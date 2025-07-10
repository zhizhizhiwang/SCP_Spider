import asyncio
import aiohttp
from collections import deque
import time
import logging
from bs4 import BeautifulSoup
import lxml # for BeautifulSoup. Don't remove
import re
from urllib.parse import urljoin, quote, urlparse
from typing import Tuple, List, Dict, Set
from dataclasses import dataclass
from libzim.writer import Creator, Item, StringProvider, Hint # type: ignore
from pathlib import Path
import traceback
import mimetypes




logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

REWRITE_PREFIX = './'

def rewrite_url(url: str, base: str) -> str:
    """
    重写URL的规则函数。
    该函数重写所有有效的URL，无论内部还是外部。
    """
    # 1. 忽略空的、data: URI或锚点链接
    if not url or url.startswith(('data:', '#', 'javascript:')):
        return url

    absolute_original_url = urljoin(base, url)
    parsed_url = urlparse(absolute_original_url)
    
    return REWRITE_PREFIX + f'{parsed_url.netloc}{parsed_url.path}{'#' if parsed_url.fragment else ''}{parsed_url.fragment}{'?' if parsed_url.query else ''}{parsed_url.query}'

def get_mimetype(url: str) -> str:
    """根据URL确定MIME类型"""
    guess = mimetypes.guess_type(url)[0]
    if guess:
        return guess

    # 默认MIME类型
    if url.endswith((".html", ".htm")):
        return "text/html"
    elif url.endswith(".css"):
        return "text/css"
    elif url.endswith(".js"):
        return "application/javascript"
    elif url.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif url.endswith(".png"):
        return "image/png"
    elif url.endswith(".gif"):
        return "image/gif"
    elif url.endswith(".svg"):
        return "image/svg+xml"
    elif url.endswith(".ico"):
        return "image/x-icon"
    else:
        return "application/octet-stream"
    
def is_resource_file(url: str) -> bool:
    """判断URL是否为资源文件（图片、CSS、JS等）"""

    return not 'html' in get_mimetype(url) 


def process_html(html_content: str, baseurl: str):
    """
    主处理函数，解析、统计和重写HTML中的所有资源。
    """
    soup = BeautifulSoup(html_content, 'lxml')
    resource_urls = set()
    new_urls = set()
    
    def add_url(url:str):
        if is_resource_file(url):
            resource_urls.add(url)
        else:
            new_urls.add(url)

    # --- 1. 处理常见的HTML标签 ---
    tag_attrs = {
        'link': ['href'], 'script': ['src'], 'img': ['src', 'srcset'],
        'audio': ['src'], 'video': ['src', 'poster'], 'source': ['src', 'srcset'],
        'iframe': ['src'], 'embed': ['src'], 'object': ['data'],
    }

    for tag_name, attrs in tag_attrs.items():
        for tag in soup.find_all(tag_name):
            for attr in attrs:
                if tag.has_attr(attr):
                    original_url_attr = tag[attr]
                    
                    if attr == 'srcset':
                        new_srcset_parts = []
                        for part in original_url_attr.split(','):
                            part = part.strip()
                            if not part: continue
                            url_part, *descriptor = part.split(None, 1)
                            add_url(url_part)
                            rewritten = rewrite_url(url_part, baseurl)
                            new_srcset_parts.append(f"{rewritten} {' '.join(descriptor)}")
                        tag[attr] = ', '.join(new_srcset_parts)
                    else:
                        add_url(original_url_attr)
                        tag[attr] = rewrite_url(original_url_attr, baseurl)

    # --- 2. 处理CSS中的url() ---
    def create_rewriter_callback(is_js=False):
        def rewriter(match):
            # JS正则可能捕获引号，CSS不会
            group_index = 2 if is_js else 1
            quote = match.group(1) if is_js else "'"
            
            original_url = match.group(group_index).strip("'\"")
            add_url(original_url)
            rewritten_url = rewrite_url(original_url, baseurl)
            
            if is_js:
                return f"{quote}{rewritten_url}{quote}"
            else:
                return f"url('{rewritten_url}')"
        return rewriter

    for style_tag in soup.find_all('style'):
        if style_tag.string:
            style_tag.string = re.sub(r'url\((.*?)\)', create_rewriter_callback(), style_tag.string)

    for tag in soup.find_all(style=True):
        tag['style'] = re.sub(r'url\((.*?)\)', create_rewriter_callback(), tag['style'])

    # --- 3. 处理JavaScript中的链接 (简单情况) ---
    for script_tag in soup.find_all('script'):
        if script_tag.string:
            # 改进正则以更好地处理简单字符串
            new_js = re.sub(r'(["\'])((?:/|https?://|./|../).*?)\1', create_rewriter_callback(is_js=True), script_tag.string)
            script_tag.string = new_js
    logging.info(f"在{baseurl}找到了{len(resource_urls)}个资源链接, {len(new_urls)}个页面链接")
    return soup.prettify(), sorted(list(resource_urls)), sorted(list(new_urls))

def process_css_content(
    css_content: str, 
    css_base_url: str, 
) -> Tuple[str, List[str]]:
    """
    处理CSS内容，重写其中的链接，并返回重写后的内容和找到的原始URL列表。

    Args:
        css_content (str): 要处理的CSS代码字符串。
        css_base_url (str): 该CSS文件的绝对URL，用于解析相对路径。

    Returns:
        Tuple[str, List[str]]: 
            - 第一个元素是重写了所有链接后的新CSS内容。
            - 第二个元素是在CSS中找到的所有原始URL（绝对路径形式）的列表。
    """
    
    resource_urls = set()
    new_urls = set()
    
    def add_url(url:str):
        if is_resource_file(url):
            resource_urls.add(url)
        else:
            new_urls.add(url)

    # 1. 处理 @import "..." 或 @import url(...)
    def import_replacer(match: re.Match) -> str:
        # group(1) 匹配 url(...) 中的内容, group(2) 匹配 "..." 中的内容
        original_url = (match.group(1) or match.group(2)).strip("'\"")
        
        # 忽略空URL
        if not original_url:
            return match.group(0)

        # 将相对URL转换为绝对URL
        absolute_url = urljoin(css_base_url, original_url)
        add_url(absolute_url)
        
        # 从 rewrite_map 中查找重写后的URL，如果找不到则保持原样
        rewritten_url = rewrite_url(original_url, css_base_url)
        
        # 返回标准格式的 @import
        return f'@import url("{rewritten_url}");'

    # 正则表达式匹配 @import 规则
    # @import url(path) or @import "path" or @import 'path'
    import_pattern = re.compile(r'@import\s+(?:url\((.*?)\)|["\'](.*?)["\']);?', re.IGNORECASE)
    modified_content = import_pattern.sub(import_replacer, css_content)

    # 2. 处理 url(...)
    def url_replacer(match: re.Match) -> str:
        original_url = match.group(1).strip("'\"")

        # 忽略空的、data: URI 或锚点
        if not original_url or original_url.startswith(('data:', '#')):
            return match.group(0)
            
        absolute_url = urljoin(css_base_url, original_url)
        add_url(absolute_url)

        rewritten_url = rewrite_url(original_url, css_base_url)
        
        # 返回标准格式的 url()
        # 注意对URL中的特殊字符（如引号）进行转义，以防破坏CSS语法
        escaped_rewritten_url = rewritten_url.replace('"', '\\"')
        return f'url("{escaped_rewritten_url}")'

    # 正则表达式匹配 url() 函数
    url_pattern = re.compile(r'url\((.*?)\)', re.IGNORECASE)
    modified_content = url_pattern.sub(url_replacer, modified_content)
    logging.info(f"在{css_base_url}找到了{len(resource_urls)}个资源链接 {len(new_urls)}个页面链接")
    return modified_content, sorted(list(resource_urls)), sorted(list(new_urls))

@dataclass
class CrawlResult:
    url: str
    remark: str
    content: str | bytes = ''
    mimetype: str = "text/html"

class AsyncHTTPFetcher:
    def __init__(self, max_concurrent=10, rate_limit=5):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit  # 每秒请求数限制
        self.queue = asyncio.Queue()
        self.active_tasks = set()
        self.session = None
        self.seen_urls = set()  # URL去重集合
        self.last_request_time = deque()  # 用于速率限制
        self.running = True
        self.results: List[CrawlResult] = []
        
    async def add_url(self, url):
        """添加新的URL到队列（带去重检查）"""
        if url not in self.seen_urls:
            self.seen_urls.add(url)
            await self.queue.put(url)
            logging.info(f"添加url:{url}")
            return True
        logging.info(f"url已经存在: {url}")
        return False

    async def add_urls(self, urls):
        """批量添加URL"""
        added = 0
        for url in urls:
            if await self.add_url(url):
                added += 1
        return added

    async def rate_limiter(self):
        """请求速率限制器"""
        now = time.monotonic()
        # 移除超过1秒的请求时间记录
        while self.last_request_time and now - self.last_request_time[0] > 1:
            self.last_request_time.popleft()
        
        # 如果达到速率限制，等待直到有空档
        if len(self.last_request_time) >= self.rate_limit:
            wait_time = 1 - (now - self.last_request_time[0])
            await asyncio.sleep(wait_time)
            now = time.monotonic()  # 更新当前时间
        
        # 添加新的请求时间
        self.last_request_time.append(now)

    async def fetch(self, url):
        """异步获取单个URL的内容"""
        try:
            await self.rate_limiter()  # 应用速率限制
            
            async with self.session.get(url, timeout=10) as response:
                content = await response.text()
                status = response.status
                logging.info(f"✅ [{status}] Fetched {url} - type: {response.content_type}")
                return url, content, status, response.content_type
        except asyncio.TimeoutError:
            logging.warning(f"⏰ 获取超时:{url}")
            return url, None, "Timeout"
        except Exception as e:
            logging.error(f"❌ 获取错误 {url}: {str(e)}")
            return url, None, str(e)

    async def worker(self):
        """工作协程，从队列获取并处理任务"""
        while self.running:
            try:
                url = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                task = asyncio.create_task(self.fetch(url))
                self.active_tasks.add(task)
                
                # 添加回调处理任务完成后的清理
                def callback(future):
                    self.active_tasks.discard(task)
                    self.queue.task_done()
                    
                    # 这里可以添加结果处理逻辑
                    result = future.result()
                    if result:
                        url, content, status, mimetype = result
                        
                        # 从内容中提取新URL并添加
                        new_urls = self.parse_links(content, mimetype, url)
                        asyncio.create_task(self.add_urls(new_urls))
                
                task.add_done_callback(callback)
            except asyncio.TimeoutError:
                # 队列空超时，继续循环
                continue

    def parse_links(self, content, mimetype, url):
        """从HTML内容中解析链接"""
        if ('html' in mimetype):
            re_content, urls = process_html(content, url)
        elif ('css' in mimetype):
            re_content, urls = process_css_content(content, url)
        else:
            re_content = content
            urls = []        
        
        self.results.append(CrawlResult(remark="success", url=url, content=re_content, mimetype=mimetype))
        return urls

    async def save_to_zim(self, output_file: str, title: str, description: str, creator_meta: str):
        try:
            zim_creator = Creator(Path(output_file))
            
            # 启动Creator上下文（所有操作必须在with块内完成）
            with zim_creator as creator:
                creator.set_mainpath("index.html")
                creator.add_metadata("Title", title)
                creator.add_metadata("Description", description)
                creator.add_metadata("Creator", creator_meta)
                creator.add_metadata("Publisher", "SCP_Spider")
                creator.add_metadata("Language", "zh")

                # 生成首页索引
                index_content = "<html><head><title>爬虫结果索引</title></head><body>"
                index_content += f"<h1>{title}</h1><p>{description}</p><hr/>"
                index_content += "<h2>已爬取的页面</h2><ul>"
                page_count = 0
                file_count = 0

                # 处理所有结果
                for result in self.results:

                    # 生成ZIM路径（与原逻辑相同）
                    '''parsed_url = urlparse(result.url)
                    path = parsed_url.path
                    if not path or path == "/":
                        path = "/index.html"
                    elif not path.endswith((".html", ".htm", ".css", ".js", ".jpg", ".png", ".gif")):
                        path = path.rstrip("/") + "/index.html"
                    if path.startswith("/"):
                        path = path[1:]
                    if '#' in result.url:
                        path = f'{path}#{result.url.split("#")[-1]}'
                    '''
                    path = result.url.replace('https://', '').replace('http://', '')


                    # 创建条目并添加
                    item = Item()
                    item.get_contentprovider = lambda: StringProvider(result.content.encode("utf-8"))
                    item.get_path = lambda: result.url
                    item.get_title = lambda: result.url
                    item.get_mimetype = lambda: result.mimetype
                    item.get_hints = lambda: {Hint.FRONT_ARTICLE: True}
                    logging.info(f'生成路径：{item.get_path()}')
                    creator.add_item(item)  # 在上下文中添加
                    
                    
                    # 更新索引
                    if result.is_resource:
                        file_count += 1
                        index_content += f'<li><a href="{item.get_path()}">{item.get_path()}</a></li>'
                    else:
                        page_count += 1
                        index_content += f'<li><a href="{path}">{result.url}</a></li>'
                    del item # 是的, 生命周期

                index_content += "</ul></body></html>"

                try:
                    # 添加索引页
                    index_content_provider = StringProvider(index_content.encode("utf-8"))
                    index_item = Item()
                    index_item.get_path = lambda: f"index.html"
                    index_item.get_title = lambda: "索引"
                    index_item.get_mimetype = lambda: "text/html"
                    index_item.get_contentprovider = lambda: index_content_provider
                    index_item.get_hints = lambda: {Hint.FRONT_ARTICLE: True}
                    creator.add_item(index_item)  # 在上下文中添加
                except Exception as e:
                    logging.error(f"索引已经存在: {e}")

            # 上下文退出时会自动完成ZIM文件写入
            logging.info(f"ZIM文件创建成功：{output_file}，包含{page_count}个页面和{file_count}个资源文件")
            return True
        except Exception as e:
            logging.error(f"创建ZIM文件失败：{e}")
            traceback.print_exc()
            return False
        
        

    async def stop(self):
        """安全停止爬虫"""
        self.running = False
        
        # 等待所有任务完成
        await self.queue.join()
        await asyncio.gather(*self.active_tasks)
        
        if self.session:
            await self.session.close()
        logging.info("🔴 爬虫进程退出")

    async def run(self, initial_urls=None):
        """启动爬虫系统"""
        self.session = aiohttp.ClientSession()
        
        # 添加初始URL
        if initial_urls:
            await self.add_urls(initial_urls)
        
        # 创建工作协程
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.max_concurrent)
        ]
        
        logging.info(f"🚀 启动 进程数:{self.max_concurrent}")
        
        # 保持主循环运行直到被停止
        while self.running:
            await asyncio.sleep(1)
            if self.queue.empty():
                self.running = False
        
        # 清理工作
        await self.stop()
        await asyncio.gather(*workers, return_exceptions=True)
        
        await self.save_to_zim(        
                output_file= f"{initial_urls[0].replace(".", "_").replace(r'/', '_')}-{time.strftime('%Y-%m-%d_%H-%M-%S')}.zim",
                title=f"{initial_urls} 爬虫存档",
                description=f"{initial_urls} 网站的离线存档, 由SCP_Spider创建",
                creator_meta="SCP_Spider",)

# 使用示例
async def main():
    fetcher = AsyncHTTPFetcher(max_concurrent=3, rate_limit=2)
    
    # 初始URL
    initial_urls = [
        'scp-wiki-cn.wikidot.com/cytus-3',
    ]
    
    # 启动爬虫
    fetcher_task = asyncio.create_task(fetcher.run(initial_urls))
    
    # 等待爬虫完全停止
    await fetcher_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")