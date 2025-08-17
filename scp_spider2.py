# playwright_archiver.py
# 一个使用Playwright动态捕获和静态内容重写的网站归档工具
import asyncio
import logging
import os
import re
import time
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from typing import Dict, Set, Any, List, Tuple
import json
from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    Page,
    Route,
    Request,
    Response,
    BrowserContext,
)
import pyjsparser

# libzim 库导入，如果未安装则提供友好提示
try:
    
    from libzim.writer import Creator, Item, StringProvider, Hint
except ImportError:
    logging.error(
        "libzim 未找到。ZIM文件保存功能将不可用。\n"
        "请通过 'pip install libzim' 安装。"
    )
    # 创建占位符，以防程序因缺少模块而崩溃
    Creator = Item = StringProvider = Hint = object


# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(module)s) %(message)s",
)


# --- 数据类定义 ---
@dataclass
class CrawlResult:
    """存储单个捕获资源的所有信息"""

    url: str
    success: bool
    status_code: int
    mimetype: str

    # content先保存为原始的bytes，在重写阶段再处理
    content: bytes
    remark: str = ""
    is_resource: bool = False


# --- 辅助函数 ---
def get_mimetype(url: str) -> str:
    """根据URL后缀猜测MIME类型，作为备用方案"""
    path = urlparse(url).path
    guess, _ = mimetypes.guess_type(path)
    if guess:
        return guess

    ext_map = {
        ".html": "text/html",
        ".htm": "text/html",
        ".css": "text/css",
        ".js": "application/javascript",
        ".json": "application/json",
        ".xml": "application/xml",
        ".svg": "image/svg+xml",
    }
    _, ext = os.path.splitext(path)
    return ext_map.get(ext.lower(), "application/octet-stream")


def is_resource_file(mimetype: str) -> bool:
    """根据MIME类型判断是否为页面（HTML）"""
    return "html" not in mimetype


def get_zim_path(url: str) -> str:
    """
    根据原始URL，生成一个干净、唯一、适合作为ZIM文件系统路径的字符串。
    格式：domain.com/path/to/resource
    """
    if not url or not url.startswith("http"):
        return ""

    parsed = urlparse(url)
    # 移除查询参数和片段
    path = parsed.path
    # ZIM路径中不包含协议头
    zim_path = f"{parsed.netloc}{path}"

    # 规范化：移除尾部斜杠（除非是根目录），并为目录添加index.html
    if len(zim_path) > 1 and zim_path.endswith("/"):
        zim_path = zim_path[:-1]
    
    # 如果路径没有文件扩展名，很可能是一个目录，为其添加index.html
    if not os.path.splitext(zim_path)[1] and not zim_path.endswith("index.html"):
        zim_path += "/index.html"
    
    # ZIM路径中的特殊字符需要被unquote
    return unquote(zim_path)


# --- 核心爬虫类 ---
class PlaywrightArchiver:
    def __init__(self, start_url: str, max_concurrent_pages: int = 5):
        self.start_url = start_url
        self.start_domain = urlparse(start_url).netloc
        self.max_concurrent_pages = max_concurrent_pages
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.processed_urls: Set[str] = set()
        self.url_map: Dict[str, str] = {}
        self.results: Dict[str, CrawlResult] = {}
        self._lock = asyncio.Lock()

    def _rewrite_content(
        self, text_content: str, content_type: str, base_url: str
    ) -> str:
        """根据内容类型，重写文本文件中的所有URL。"""

        def url_replacer_factory(original_url: str) -> str:
            """核心的URL查找和替换逻辑"""
            if not original_url or not isinstance(original_url, str): return original_url
            absolute_url = urljoin(base_url, original_url)
            return self.url_map.get(absolute_url, original_url)

        # --- HTML 和 CSS 的重写逻辑保持不变，它们工作得很好 ---
        if "html" in content_type:
            soup = BeautifulSoup(text_content, "lxml")
            tags_to_rewrite = {
                "a": "href", "link": "href", "script": "src", "img": "src",
                "audio": "src", "video": "src", "source": "src", "iframe": "src",
                "embed": "src", "object": "data", "form": "action",
            }
            for tag_name, attr in tags_to_rewrite.items():
                for tag in soup.find_all(tag_name, **{attr: True}):
                    tag[attr] = url_replacer_factory(tag[attr])
            for tag in soup.find_all(srcset=True):
                 parts = [p.strip().split() for p in tag['srcset'].split(',')]
                 new_srcset = ", ".join([f"{url_replacer_factory(p[0])} {' '.join(p[1:])}" for p in parts if p])
                 tag['srcset'] = new_srcset
            return soup.prettify()

        if "css" in content_type:
            pattern = re.compile(r"url\((.*?)\)", re.IGNORECASE)
            def css_replacer(match: re.Match) -> str:
                original = match.group(1).strip("'\"")
                return f"url('{url_replacer_factory(original)}')"
            return pattern.sub(css_replacer, text_content)

        # --- 新的 JavaScript 重写逻辑 ---
        if "javascript" in content_type:
            replacements: List[Tuple[int, int, str]] = []

            def visit_node(node):
                """递归遍历AST节点"""
                if not isinstance(node, dict): return
                
                # 检查当前节点是否是字符串字面量
                if node.get('type') == 'Literal' and isinstance(node.get('value'), str):
                    original_str = node['value']
                    # 启发式判断是否为URL
                    if '/' in original_str or ('.' in original_str and ' ' not in original_str):
                        rewritten_url = url_replacer_factory(original_str)
                        if rewritten_url != original_str:
                            # 记录替换信息：(起始索引, 结束索引, 新的带引号的字符串)
                            # pyjsparser的 'range' 包含了字符串引号，非常方便
                            start, end = node['range']
                            # 需要将新URL封装在引号里
                            new_quoted_str = f'"{rewritten_url}"' 
                            replacements.append((start, end, new_quoted_str))

                # 递归遍历子节点
                for key, value in node.items():
                    if isinstance(value, dict):
                        visit_node(value)
                    elif isinstance(value, list):
                        for item in value:
                            visit_node(item)
            
            try:
                # 1. 解析JS，带上range信息
                ast = pyjsparser.parse(text_content, {'range': True})
                # 2. 遍历AST，收集所有需要替换的位置
                visit_node(ast)
                
                # 3. 如果有需要替换的内容，执行替换
                if replacements:
                    # 从后往前替换，避免索引失效
                    replacements.sort(key=lambda x: x[0], reverse=True)
                    new_js_parts = []
                    last_index = len(text_content)
                    for start, end, new_str in replacements:
                        # 添加未被替换的后半部分
                        new_js_parts.append(text_content[end:last_index])
                        # 添加替换后的新字符串
                        new_js_parts.append(new_str)
                        last_index = start
                    # 添加最开始未被替换的部分
                    new_js_parts.append(text_content[:last_index])
                    # 翻转并拼接
                    return "".join(reversed(new_js_parts))
                else:
                    return text_content # 没有需要替换的，返回原文
            except Exception as e:
                logging.warning(f"pyjsparser解析JS失败({base_url}): {e}. 回退到正则。")
                # 正则表达式作为后备方案
                pattern = re.compile(r"""(["'])((?:/|https?://|\./|\.\./)[^"']+)\1""")
                def js_replacer(match: re.Match) -> str:
                    quote, original = match.groups()
                    return f"{quote}{url_replacer_factory(original)}{quote}"
                return pattern.sub(js_replacer, text_content)

        return text_content

    async def _handle_response(self, response: Response):
        """
        Playwright的响应处理器，在每次浏览器接收到响应时触发。
        这是捕获阶段的核心。
        """
        url = response.url
        # 1. 过滤：只处理目标域名下的、http/https协议的URL
        if not is_resource_file(url) and (not url.startswith("http") or urlparse(url).netloc != self.start_domain):
            logging.info(f"忽略非资源跨域外链: {url}")
            return
        
        # 2. 过滤：只处理目标路径下的URL
        if not is_resource_file(url) and not urlparse(url).path.startswith(urlparse(self.start_url).path):
            logging.info(f"忽略非资源外链: {url}")
            return

        async with self._lock:
            # 防止重复处理同一个URL
            if url in self.processed_urls:
                return
            self.processed_urls.add(url)
            
        zim_path = get_zim_path(url)
        if not zim_path:
            return
            
        logging.info(f"捕获到响应: {url} ({response.status})")

        # 2. 建立映射：将原始URL与其ZIM路径关联起来
        async with self._lock:
            self.url_map[url] = zim_path

        # 3. 获取响应内容并保存
        try:
            # 对于重定向，Playwright会自动处理，我们只需记录原始URL的映射即可
            if 300 <= response.status < 400:
                return
            if response.status >= 400:
                logging.warning(f"请求失败 {response.status}: {url}")
                return

            body = await response.body()
            mimetype = response.headers.get("content-type", "").split(";")[0] or get_mimetype(url)
            
            result = CrawlResult(
                url=url,
                success=True,
                status_code=response.status,
                mimetype=mimetype,
                content=body,
                is_resource=is_resource_file(mimetype),
            )
            # 4. 存入结果字典
            async with self._lock:
                self.results[zim_path] = result

        except Exception as e:
            logging.error(f"处理响应失败 ({url}): {e}")
            
    async def _discover_links_and_queue(self, page: Page):
        """在一个页面上查找所有<a>标签的链接，并加入队列"""
        links = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
        async with self._lock:
            for link in links:
                # 过滤并加入队列
                if link and link.startswith("http") and urlparse(link).netloc == self.start_domain:
                    if link not in self.processed_urls and link not in list(self.queue._queue):
                        await self.queue.put(link)

    async def _worker(self, context: BrowserContext):
        """单个工作协程，负责从队列中取URL并访问"""
        page = await context.new_page()
        while True:
            try:
                url_to_process = await self.queue.get()
                logging.info(f"Worker开始处理: {url_to_process}")
                await page.goto(url_to_process, wait_until="networkidle", timeout=60000)
                # 访问后，再次探索页面上的静态链接，以防万一
                await self._discover_links_and_queue(page)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Worker处理页面失败 ({url_to_process}): {e}")
            finally:
                self.queue.task_done()
        await page.close()


    async def crawl(self):
        """主爬取流程，协调所有工作"""
        await self.queue.put(self.start_url)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(
                java_script_enabled=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36"
            )
            # 全局响应处理器，捕获所有请求
            context.on("response", self._handle_response)
            
            # 创建一组并发的工作者
            tasks = [asyncio.create_task(self._worker(context)) for _ in range(self.max_concurrent_pages)]

            # 等待队列处理完毕
            await self.queue.join()
            
            # 所有任务完成，取消工作者
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            await context.close()
            await browser.close()

        # --- 阶段二：内容重写 ---
        logging.info(f"捕获完成，共{len(self.results)}个资源。开始内容重写...")
        relative_url_map = {
            original_url: f"/{zim_path}" 
            for original_url, zim_path in self.url_map.items()
        }
        url_map_json = json.dumps(relative_url_map, indent=2)

        # 2. 读取我们的Service Worker模板
        try:
            with open('sw_template.js', 'r', encoding='utf-8') as f:
                sw_template = f.read()
            # 将URL映射注入到SW模板中
            self.sw_script_content = sw_template.replace(
                'const URL_MAP = {};', 
                f'const URL_MAP = {url_map_json};'
            )
        except FileNotFoundError:
            logging.error("sw_template.js 未找到！Service Worker无法被注入。")
            self.sw_script_content = ""

        # 3. 遍历所有结果，进行重写和注入
        for zim_path, result in self.results.items():
            content_type = result.mimetype.lower()
            is_text = "text" in content_type or "javascript" in content_type or "json" in content_type or "xml" in content_type
            
            if result.content and is_text:
                try:
                    text_content = result.content.decode("utf-8")
                    
                    # 首先，进行常规的静态URL重写
                    rewritten_content = self._rewrite_content(
                        text_content, content_type, result.url
                    )

                    # 如果是HTML页面，额外注入Service Worker注册脚本
                    if "html" in content_type and self.sw_script_content:
                        soup = BeautifulSoup(rewritten_content, "lxml")
                        # 创建注入脚本
                        sw_registration_script = soup.new_tag("script")
                        sw_registration_script.string = """
                        if ('serviceWorker' in navigator) {
                            navigator.serviceWorker.register('/sw.js')
                                .then(registration => console.log('ServiceWorker registration successful with scope: ', registration.scope))
                                .catch(err => console.log('ServiceWorker registration failed: ', err));
                        }
                        """
                        # 将脚本添加到body末尾
                        if soup.body:
                            soup.body.append(sw_registration_script)
                        else:
                            soup.append(sw_registration_script) # 如果没有body
                        
                        rewritten_content = str(soup)

                    result.content = rewritten_content.encode("utf-8")
                except UnicodeDecodeError:
                    logging.warning(f"解码失败，跳过重写: {result.url}")

        logging.info("内容重写和注入完成。")

    def save_to_zim(
        self, output_file: str, title: str, description: str, creator: str
    ):
        """将处理好的结果，连同我们自己的sw.js，一起保存为ZIM文件"""
        logging.info(f"正在保存到 ZIM 文件: {output_file}")
        if not globals().get('Creator'):
            logging.error("libzim 未加载，无法保存 ZIM 文件。")
            return

        try:
            with Creator(Path(output_file)) as zim_creator:
                # ... (元数据设置与之前相同)
                main_page_zim_path = get_zim_path(self.start_url)
                zim_creator.set_mainpath(main_page_zim_path)
                # ...

                # 1. 添加所有捕获到的资源
                for zim_path, result in self.results.items():
                    if not result.success: continue
                    item = Item(
                        path=zim_path,
                        title=result.url,
                        mimetype=result.mimetype,
                        contentprovider=StringProvider(result.content),
                        hints={Hint.COMPRESS: True},
                    )
                    zim_creator.add_item(item)

                # 2. 关键：添加我们自己生成的Service Worker文件
                if hasattr(self, 'sw_script_content') and self.sw_script_content:
                    sw_item = Item(
                        path="sw.js", # 必须是根目录下的sw.js
                        title="Service Worker",
                        mimetype="application/javascript",
                        contentprovider=StringProvider(self.sw_script_content.encode('utf-8')),
                        hints={Hint.COMPRESS: True}
                    )
                    zim_creator.add_item(sw_item)
                    logging.info("自定义 Service Worker (sw.js) 已添加到 ZIM 文件。")

            logging.info(f"ZIM 文件创建成功: {output_file}")
        except Exception as e:
            logging.error(f"创建ZIM文件失败: {e}", exc_info=True)

async def main():
    """主执行函数"""
    # --- 配置区 ---
    START_URL = "https://scp-wiki-cn.wikidot.com/scp-cn-3959"
    # START_URL = "https://www.douban.com/" # 也可以试试其他复杂的网站
    CONCURRENT_PAGES = 5  # 并发打开的标签页数量，根据你的机器性能调整
    # --- 配置区结束 ---

    archiver = PlaywrightArchiver(START_URL, CONCURRENT_PAGES)
    
    # 阶段一：抓取与捕获
    await archiver.crawl()

    # 阶段二：保存
    domain = urlparse(START_URL).netloc
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    zim_filename = f"{domain.replace('.', '_')}_{timestamp}.zim"
    
    archiver.save_to_zim(
        output_file=zim_filename,
        title=f"Archive of {domain}",
        description=f"An offline archive of {START_URL} created on {timestamp}",
        creator="Playwright Archiver",
    )


if __name__ == "__main__":
    # 确保你已经安装了必要的依赖:
    # pip install playwright beautifulsoup4 slimit lxml libzim
    # playwright install chromium
    asyncio.run(main())