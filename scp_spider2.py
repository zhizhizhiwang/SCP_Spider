# playwright_archiver.py
# 一个使用Playwright动态捕获和静态内容重写的网站归档工具
import asyncio
import logging
import colorlog
import os
import re
import time
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from typing import Dict, Set, Any, List, Tuple
import json
import aiohttp
from typing import Optional
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
import threading

def get_logger(level=logging.INFO):
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(level)
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # 定义颜色输出格式
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s(%(asctime)s) - [%(threadName)s] - :\n[%(levelname)s]%(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    formatter = logging.Formatter('(%(asctime)s)-[%(threadName)s]-: [%(levelname)s]%(message)s')
    
    # 将颜色输出格式添加到控制台日志处理器
    console_handler.setFormatter(color_formatter)
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # 将控制台日志处理器添加到logger对象
    logger.addHandler(console_handler)
    
    # 创建 FileHandler 对象
    file_handler = logging.FileHandler('spider.log', mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    
    # 将 FileHandler 添加到 logger 对象
    logger.addHandler(file_handler)
    
    return logger

logger = get_logger(level=logging.DEBUG)

# libzim 库导入，如果未安装则提供友好提示
try:
    from libzim.writer import Creator, Item, StringProvider, Hint  # type: ignore
except ImportError:
    logger.error(
        "libzim 未找到。ZIM文件保存功能将不可用。\n"
        "请通过 'pip install libzim' 安装。"
    )
    raise ImportError("libzim 未找到。")

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



# --- 全局变量 ---
ONE_PAGE_MOD = False
MAX_FETCH_PAGES = 1000

def setMAX_FETCH_PAGES(num: int):
    global MAX_FETCH_PAGES
    MAX_FETCH_PAGES = num

def getMAX_FETCH_PAGES() -> int:
    global MAX_FETCH_PAGES
    return MAX_FETCH_PAGES

def setONE_PAGE_MOD(bool: bool):
    global ONE_PAGE_MOD
    ONE_PAGE_MOD = bool

def getONE_PAGE_MOD() -> bool:
    global ONE_PAGE_MOD
    return ONE_PAGE_MOD

# --- 辅助函数 ---
async def get_mimetype(url: str, session: Optional[aiohttp.ClientSession] = None) -> str:
    """
    通过请求头和URL后缀来确定MIME类型
    
    Args:
        url: 要检测的URL
        session: 可选的aiohttp会话对象，如果不提供则创建新的

    Returns:
        str: MIME类型字符串
    """
    # 1. 首先尝试从URL后缀猜测
    path = urlparse(url).path
    guess, _ = mimetypes.guess_type(path)
    if guess:
        return guess

    '''
        # 2. 尝试发送HEAD请求获取Content-Type
        if url.startswith(('http://', 'https://')):
            try:
                should_close = session is None
                session = session or aiohttp.ClientSession()
                
                async with session.head(url, timeout=5) as response:
                    if content_type := response.headers.get('Content-Type', '').split(';')[0]:
                        return content_type
                        
            except Exception as e:
                logger.warning(f"获取MIME类型时出错 ({url}): {e}")
            finally:
                if should_close and session:
                    await session.close()
    '''
    
    # 3. 如果上述方法都失败，使用扩展名映射表
    ext_map = {
        ".html": "text/html",
        ".htm": "text/html", 
        ".css": "text/css",
        ".js": "application/javascript",
        ".json": "application/json",
        ".xml": "application/xml",
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".ttf": "font/ttf",
        ".php": "text/html",
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
    
    # ZIM路径中的特殊字符需要被unquote
    return unquote(zim_path)

def normalize_url(raw_url: str, base: Optional[str] = None) -> str:
    """
    标准化 URL，用于去重与队列判断：
    - 解析相对URL（如果给了 base）
    - 去除 fragment（#...）
    - 合并重复斜杠，移除末尾多余的 '/'
    - 统一 scheme + netloc（小写）
    - 保留查询参数（不排序，若需要可扩展）
    """
    if not raw_url:
        return ""

    # 解析相对 URL
    if base and not raw_url.startswith(("http://", "https://")):
        raw_url = urljoin(base, raw_url)

    parsed = urlparse(raw_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()

    # 合并重复斜杠并规范 path
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query = parsed.query or ""
    # 丢弃 fragment
    if query:
        return f"{scheme}://{netloc}{path}?{query}"
    else:
        return f"{scheme}://{netloc}{path}"

def get_std_link_path(url: str) -> str:
    """
    返回标准化的可用于比较/入队的链接字符串（会去掉 fragment 等）。
    """
    return normalize_url(url)

'''
sw_item = Item(
                        path="https://www.sw.com/sw.js", # 必须是根目录下的sw.js
                        title="Service Worker",
                        mimetype="application/javascript",
                        contentprovider=StringProvider(self.sw_script_content.encode('utf-8')),
                        hints={Hint.COMPRESS: True}
                    )
'''
def create_zim_item(path: str, title: str, mimetype: str, contentprovider: StringProvider, hints: Dict[Hint, Any]) -> Item:
    item = Item()
    item.get_path = lambda: path
    item.get_title = lambda: title
    item.get_mimetype = lambda: mimetype
    item.get_contentprovider = lambda: contentprovider
    item.get_hints = lambda: hints
    return item
    

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
        self.pages_num = 0

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
                ast = pyjsparser.parse(text_content)
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
                logger.warning(f"pyjsparser解析JS失败({base_url}): {e}. 回退到正则。")
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
        if not (url.startswith(('http://', 'https://'))):
            logger.info(f"忽略非http/https资源: {url}")
            return
        
        # 被 handle 捕获的一律视为资源
        """ 
        # 2. 过滤：只处理目标路径下的URL
        if not is_resource_file(await get_mimetype(url)) and not urlparse(url).path.startswith(urlparse(self.start_url).path):
            logger.info(f"忽略非资源外链: {url}")
            return
        
        if getONE_PAGE_MOD() and not is_resource_file(await get_mimetype(url)) and urlparse(url).netloc == self.start_domain and urlparse(url).path != urlparse(self.start_url).path:
            logger.info(f"忽略非主页面外链: {url}")
            return
        """

        async with self._lock:
            # 防止重复处理同一个URL
            if get_std_link_path(url) in self.processed_urls:
                return
            self.processed_urls.add(get_std_link_path(url))
            
        zim_path = get_zim_path(url)
        if not zim_path:
            return
            
        logger.info(f"[当前待处理:{len(self.processed_urls)}] 捕获到资源响应: {url} ({response.status})")

        # 2. 建立映射：将原始URL与其ZIM路径关联起来
        async with self._lock:
            self.url_map[url] = zim_path

        # 3. 获取响应内容并保存
        try:
            # 对于重定向，Playwright会自动处理，我们只需记录原始URL的映射即可
            if 300 <= response.status < 400:
                return
            if response.status >= 400:
                logger.warning(f"请求失败 {response.status}: {url}")
                return

            body = await response.body()
            mimetype = response.headers.get("content-type", "").split(";")[0] or await get_mimetype(url)
            
            result = CrawlResult(
                url=url,
                success=True,
                status_code=response.status,
                mimetype=mimetype,
                content=body,
                is_resource=False,
            )
            # 4. 存入结果字典
            async with self._lock:
                self.results[zim_path] = result

        except Exception as e:
            logger.error(f"处理响应失败 ({url}): {e}")
            
    async def _discover_links_and_queue(self, page: Page):
        """在一个页面上查找所有<a>标签的链接，并加入队列"""
        discovered_links = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
        async with self._lock:
            for raw_link in discovered_links:
                norm = normalize_url(raw_link, base=page.url)
                if not norm:
                    continue
                # 去重：已处理或已在队列中则跳过
                if norm in self.processed_urls:
                    logger.debug(f"已处理过，跳过: {norm}")
                    continue
                # 可选：检查队列内是否已存在（简单扫描）
                in_queue = False
                try:
                    q_items = list(self.queue._queue)  # 仅在调试/小规模时可用
                    if norm in q_items:
                        in_queue = True
                except Exception:
                    in_queue = False

                if in_queue:
                    logger.debug(f"已在队列中，跳过: {norm}")
                    continue
                
                if self.pages_num >= getMAX_FETCH_PAGES():
                    logger.info(f"已达到最大抓取页面数({getMAX_FETCH_PAGES()})，停止抓取。")
                    continue
                
                if getONE_PAGE_MOD() and (urlparse(norm).netloc != self.start_domain or urlparse(norm).path != urlparse(self.start_url).path):
                    logger.info(f"(启用单页模式)忽略非主页面外链: {norm}")
                    continue
                

                await self.queue.put(norm)
                logger.info(f"[当前任务数:{self.queue.qsize()}]发现新链接: {norm}")

    async def _worker(self, context: BrowserContext):
        """单个工作协程，负责从队列中取URL并访问"""
        page = await context.new_page()
        while True:
            try:
                url_to_process = await self.queue.get()
                logger.info(f"[剩余: {self.queue.qsize()}] Worker开始处理: {url_to_process}")
                await page.goto(url_to_process, wait_until="networkidle", timeout=60000)
                # 访问后，再次探索页面上的静态链接，以防万一
                await self._discover_links_and_queue(page)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker处理页面失败 ({url_to_process}): {e}")
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
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36 SCP-Archiver/1.0"
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
        logger.info(f"捕获完成，共{len(self.results)}个资源。开始内容重写...")
        
        relative_url_map = {
            original_url: f"/{zim_path}" 
            for original_url, zim_path in self.url_map.items()
        }
        url_map_json = json.dumps(relative_url_map, indent=2)
        logger.debug(f"URL映射表: {url_map_json}")

        # 2. 读取我们的 Interceptor 模板
        try:
            # 文件名改为 url_interceptor.js
            with open('url_interceptor.js', 'r', encoding='utf-8') as f:
                interceptor_template = f.read()
            # 将URL映射注入到模板中
            self.interceptor_script_content = interceptor_template.replace(
                'const URL_MAP = {};', 
                f'const URL_MAP = {url_map_json};'
            )
        except FileNotFoundError:
            logger.error("url_interceptor.js 未找到！URL拦截器无法被注入。")
            self.interceptor_script_content = ""

        # 3. 遍历所有结果，进行重写和注入
        for zim_path, result in self.results.items():
            content_type = result.mimetype.lower()
            is_text = "text" in content_type or "javascript" in content_type or "json" in content_type or "xml" in content_type
            
            if result.content and is_text:
                try:
                    text_content = result.content.decode("utf-8")
                    
                    
                    rewritten_content = self._rewrite_content(
                        text_content, content_type, result.url
                    )
                    

                    # 如果是HTML页面，额外注入Service Worker注册脚本
                    if "html" in content_type and self.interceptor_script_content:
                        soup = BeautifulSoup(rewritten_content, "lxml")
                        
                        interceptor_script_tag = soup.new_tag("script")
                        # 脚本路径现在是 /url_interceptor.js
                        interceptor_script_tag['src'] = '/url_interceptor.js'
                        
                        # 注入到 <head> 的最前面，确保它在其他脚本执行前运行
                        if soup.head:
                            soup.head.insert(0, interceptor_script_tag)
                        else:
                            # 如果没有head, 作为body的第一个子元素
                            if soup.body:
                                soup.body.insert(0, interceptor_script_tag)
                            else:
                                soup.insert(0, interceptor_script_tag)

                        final_content = str(soup)
                        logger.debug(f'{zim_path} 注入 URL Interceptor 完成')
                    else:
                        final_content = rewritten_content

                    result.content = final_content.encode("utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"解码失败，跳过重写: {result.url}")

        logger.info("内容重写和注入完成。")

    def save_to_zim(
        self, output_file: str, title: str, description: str, creator: str
    ):
        """将处理好的结果，连同我们自己的sw.js，一起保存为ZIM文件"""
        logger.info(f"正在保存到 ZIM 文件: {output_file}")
        if not globals().get('Creator'):
            logger.error("libzim 未加载，无法保存 ZIM 文件。")
            return

        try:
            with Creator(Path(output_file)) as zim_creator:
                main_page_zim_path = get_zim_path(self.start_url)
                zim_creator.set_mainpath(main_page_zim_path)
                zim_creator.add_metadata("Title", title)
                zim_creator.add_metadata("Description", description)
                zim_creator.add_metadata("Publisher", "SCP_Spider")
                zim_creator.add_metadata("Language", "zh")
                zim_creator.add_metadata("creator", creator)

                # 1. 添加所有捕获到的资源
                for zim_path, result in self.results.items():
                    
                    if not result.success: continue
                    item = create_zim_item(
                        path=zim_path,
                        title=result.url,
                        mimetype=result.mimetype,
                        contentprovider=StringProvider(result.content),
                        hints={Hint.COMPRESS: True},
                    )
                    zim_creator.add_item(item)
                    logger.info(f"添加资源: {zim_path} ({result.mimetype})")

                # 2. 关键：添加我们自己生成的 URL Interceptor 文件
                if hasattr(self, 'interceptor_script_content') and self.interceptor_script_content:
                    interceptor_item = create_zim_item(
                        path="url_interceptor.js", # ZIM 路径
                        title="URL Interceptor",
                        mimetype="application/javascript",
                        contentprovider=StringProvider(self.interceptor_script_content.encode('utf-8')),
                        hints={Hint.COMPRESS: True}
                    )
                    zim_creator.add_item(interceptor_item)
                    logger.info("自定义 URL Interceptor (url_interceptor.js) 已添加到 ZIM 文件。")
            
            logger.info(f"ZIM 文件创建成功: {output_file}")
        except Exception as e:
            logger.error(f"创建ZIM文件失败: {e}", exc_info=True)

async def main():
    """主执行函数"""
    # --- 配置区 ---
    START_URL = "https://scp-wiki-cn.wikidot.com/scp-cn-3959"
    # START_URL = "https://www.douban.com/" # 也可以试试其他复杂的网站
    CONCURRENT_PAGES = 5  # 并发打开的标签页数量，根据你的机器性能调整
    
    setONE_PAGE_MOD(True) # 强制抛弃所有非同域页面
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