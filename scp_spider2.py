import asyncio
import os
import mimetypes
from asyncio import sleep
import httpx
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup
import logging
import json
from datetime import datetime
from typing import Any
from dataclasses import dataclass, asdict, field
import csv
from pathlib import Path
import time
import re
import traceback
from typing import Tuple, List, Dict, Set


# libzim 相关导入
from libzim.writer import (  # pyright: ignore[reportMissingModuleSource]
    Creator,
    Item,
    StringProvider,
    Hint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
    # logging.info(f'判定{url}类型为{['资源', '页面']['html' in get_mimetype(url)]}')
    return not 'html' in get_mimetype(url) 



@dataclass
class CrawlResult:
    url: str
    success: bool
    remark: str
    status_code: int
    content: str | bytes = ''
    mimetype: str = "text/html"
    is_resource: bool = False
    
REWRITE_PREFIX = ''

def rewrite_url(url: str, base: str) -> str:
    """
    重写URL的规则函数。
    该函数重写所有有效的URL，无论内部还是外部。
    """
    # 1. 忽略空的、data: URI或锚点链接
    if not url or url.startswith(('data:', '#', 'javascript:')):
        return url
    # scp-wiki-cn.wikidot.com/cytus-3 的base环境实际上是 scp-wiki-cn.wikidot.com, 需要回退一个path层
    real_base = '/'.join(base.split('/')[0:-1])
    # logging.info(f'base环境是{real_base}')

    absolute_original_url = urljoin(base, url)
    parsed_url = urlparse(os.path.relpath(absolute_original_url, real_base).replace('\\', '/'))
    
    return REWRITE_PREFIX + f'{parsed_url.netloc}{parsed_url.path}{'#' if parsed_url.fragment else ''}{parsed_url.fragment}{'?' if parsed_url.query else ''}{parsed_url.query}'



class WebCrawler:
    def __init__(
        self,
        searching_domain: str,
        max_retries: int = 3,
        waiting_seconds_if_error: float = 3.0,
        max_concurrent: int = 10,
        store_response_text: bool = True,
        fetch_resources: bool = True,
        store_resource_content: bool = True,
    ):
        self.searching_domain = searching_domain
        self.max_retries = max_retries
        self.waiting_seconds_if_error = waiting_seconds_if_error
        self.max_concurrent = max_concurrent
        self.store_response_text = store_response_text
        self.fetch_resources = fetch_resources
        self.store_resource_content = store_resource_content

        # 设置需要跳过的文件扩展名（移除了CSS、JS和图片文件）
        self.skipped_extensions = (
            ".mp4",
            ".doc",
            ".docx",
            ".pdf",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".rar",
            ".7z",
            ".exe",
            ".apk",
            ".ipa",
            ".iso",
            ".dmg",
            ".msi",
            ".deb",
            ".rpm",
            ".tgz",
            ".bz2",
            ".gz",
            ".xz",
            ".zst",
            ".lzma",
            ".lzo",
            ".lz4",
            ".lz",
            ".z",
            ".lzma2",
            ".zstd",
            ".zpaq",
            ".arc",
            ".arj",
            ".cab",
            ".chm",
            ".cso",
            ".cpio",
            ".cramfs",
            ".fat",
            ".hfs",
            ".hfsx",
        )

        self.target_urls: set[str] = {f"https://{searching_domain}"}
        self.start_domain = urlparse((f'https://{searching_domain}'))
        self.accessed_urls: set[str] = set()
        self.results: list[CrawlResult] = []
        self.resource_urls: set[str] = set()  # 存储需要获取的资源URL
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._url_lock = asyncio.Lock()
        self.url_map : dict[str, str] = {} # 资源的url映射

    def process_html(self, html_content: str, baseurl: str):
        """
        主处理函数，解析、统计和重写HTML中的所有资源。
        """
        soup = BeautifulSoup(html_content, 'lxml')
        resource_urls = set()
        new_urls = set()
        
        def add_url(url:str):
            if is_resource_file(url):
                resource_urls.add(urljoin(baseurl, url))
            else:
                new_urls.add(urljoin(baseurl, url))

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
        return soup.prettify(), resource_urls, new_urls

    def process_css_content(
        self,
        css_content: str, 
        css_base_url: str, 
    ) -> Tuple[str, Set[str], Set[str]]:
        """
        处理CSS内容，重写其中的链接，并返回重写后的内容和找到的原始URL列表。

        Args:
            css_content (str): 要处理的CSS代码字符串。
            css_base_url (str): 该CSS文件的绝对URL，用于解析相对路径。

        Returns:
            Tuple[str, Set[str], Set[str]]: 
                - 第一个元素是重写了所有链接后的新CSS内容。
                - 第二个元素是在CSS中找到的所有资源URL（绝对路径形式）的列表。
                - 第三个元素是在CSS中找到的所有页面URL（绝对路径形式）的列表。
        """
        
        resource_urls = set()
        new_urls = set()
        
        def add_url(url:str):
            if is_resource_file(url):
                resource_urls.add(urljoin(css_base_url, url))
            else:
                new_urls.add(urljoin(css_base_url, url))

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

        return modified_content, resource_urls, new_urls

    async def _process_single_url(
        self, client: httpx.AsyncClient, url: str, is_resource: bool = False
    ) -> None:
        """处理单个URL的异步方法"""
        async with self.semaphore:
            # 检查是否已访问
            async with self._url_lock:
                if url in self.accessed_urls:
                    return
                self.accessed_urls.add(url)

            if not is_resource:  # 只对网页做扩展名过滤
                parsed_url = urlparse(url)
                if os.path.splitext(parsed_url.path)[-1] in self.skipped_extensions:
                    logging.info(f"跳过链接{url}")
                    return

            error = 0
            response = None

            while error < self.max_retries:
                try:
                    response = await client.get(url, follow_redirects=True)
                    break
                except Exception as e:
                    error += 1
                    logging.error(
                        f"Error accessing {url}: {e.__class__.__name__}, attempt {error}/{self.max_retries}"
                    )

                    if error >= self.max_retries:
                        if url.startswith("https://") and not is_resource:
                            http_url = f"http://{url.removeprefix('https://')}"
                            async with self._url_lock:
                                if http_url not in self.accessed_urls:
                                    self.target_urls.add(http_url)
                            return
                        else:
                            # 记录最终失败
                            result = CrawlResult(
                                url=url,
                                success=False,
                                remark=(
                                    "bad domain" if is_resource else "connection error"
                                ),
                                status_code=0,
                                is_resource=is_resource,
                            )
                            async with self._url_lock:
                                self.results.append(result)
                            return
                    else:
                        await asyncio.sleep(self.waiting_seconds_if_error)
            if response:
                # 获取MIME类型
                content_type = response.headers.get("content-type", "")
                mimetype = (
                    content_type.split(";")[0] if content_type else get_mimetype(url)
                )

                # 记录成功结果
                remark = (
                    "No HTTPS" if url.startswith("http://") else response.reason_phrase
                )

                # 资源文件和HTML页面的处理方式不同
                if is_resource:
                    # 资源文件保存二进制内容
                    result = CrawlResult(
                        url=url.replace('https://', '').replace('http://', ''),
                        success=response.is_success,
                        remark=remark,
                        status_code=response.status_code,
                        content=(
                            response.content if self.store_resource_content else ""
                        ),
                        mimetype=mimetype,
                        is_resource=True,
                    )
                    
                    if(mimetype == 'text/css' or 'javascript' in mimetype):
                        re_comment, new_resource_urls, new_urls = self.process_css_content(response.text, url)
                        async with self._url_lock:
                            # 只添加未访问的URL
                            self.target_urls.update(new_urls - self.accessed_urls)
                            # 添加新的资源URL
                            self.resource_urls.update(
                                new_resource_urls - self.accessed_urls
                            )
                        result.content = re_comment
                else:

                    # 提取新URL和资源URL
                    re_comment, new_resource_urls, new_urls = self.process_html(response.text, url)
                    
                    result = CrawlResult(
                        url=url,
                        success=response.is_success,
                        remark=remark,
                        status_code=response.status_code,
                        content=re_comment,  
                        mimetype=mimetype,
                        is_resource=False,
                    )
                    async with self._url_lock:
                        # 只添加未访问的URL
                        self.target_urls.update(new_urls - self.accessed_urls)
                        # 添加新的资源URL
                        self.resource_urls.update(
                            new_resource_urls - self.accessed_urls
                        )
                        
                async with self._url_lock:
                    self.results.append(result)

    async def crawl(self) -> list[dict[str, Any]]:
        """主要的爬虫方法"""
        async with httpx.AsyncClient() as client:
            # 第一阶段：爬取网页内容
            while True:
                # 获取当前批次的URL
                async with self._url_lock:
                    current_batch = list(self.target_urls - self.accessed_urls)
                    if not current_batch:
                        break

                # 批量处理URL，提高并发性
                batch_size = min(len(current_batch), self.max_concurrent)
                tasks = [
                    self._process_single_url(client, url, is_resource=False)
                    for url in current_batch[:batch_size]
                ]

                # 等待当前批次完成
                await asyncio.gather(*tasks, return_exceptions=True)

                # 记录进度
                async with self._url_lock:
                    total_urls = len(self.target_urls) + len(self.accessed_urls)
                    progress = (
                        100 * len(self.accessed_urls) / total_urls
                        if total_urls > 0
                        else 0
                    )
                    logging.info(
                        f"URLs: {progress:.2f}%, {len(self.accessed_urls)}/{total_urls}"
                    )

            # 第二阶段：爬取资源文件
            if self.fetch_resources:
                logging.info(
                    f"Starting to fetch resource files: {len(self.resource_urls)} files"
                )
                resource_count = 0

                # 批量处理资源URL
                while True:
                    async with self._url_lock:
                        current_batch = list(self.resource_urls - self.accessed_urls)
                        if not current_batch:
                            break

                    batch_size = min(len(current_batch), self.max_concurrent)
                    tasks = [
                        self._process_single_url(client, url, is_resource=True)
                        for url in current_batch[:batch_size]
                    ]

                    await asyncio.gather(*tasks, return_exceptions=True)

                    # 记录资源获取进度
                    resource_count += batch_size
                    logging.info(
                        f"Resources: {resource_count}/{len(self.resource_urls)} files"
                    )

        logging.info("Finished crawling")
        return [asdict(result) for result in self.results]

    def save_to_zim(
        self, output_file: str, title: str, description: str, creator_meta: str
    ) -> bool:
        """保存结果为ZIM格式

        Args:
            output_file: 输出的ZIM文件路径
            title: ZIM文件标题
            description: ZIM文件描述
            creator_meta: 创建者信息

        Returns:
            bool: 保存是否成功
        """

        try:
            
            zim_creator = Creator(Path(output_file))
            
            # 启动Creator上下文（所有操作必须在with块内完成）
            with zim_creator as creator:
                # === 必须在with块内设置元数据 ===
                creator.set_mainpath("index.html")
                creator.add_metadata("Title", title)
                creator.add_metadata("Description", description)
                creator.add_metadata("Creator", creator_meta)  # 避免变量名冲突
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
                    if not result.success:
                        logging.warning(
                            f"请求失败 {result.url}: {result.remark}"
                        )
                        continue

                    # 生成ZIM路径（与原逻辑相同）
                    # parsed_url = urlparse(result.url)
                    path = result.url.replace('https://', '').replace('http://', '')
                    '''
                    @https://www.openzim.org/wiki/ZIM_file_format path章节
                    path中的特殊字符保持原样
                    #和?进行转义
                    '''
                    path = unquote(path).replace('#', '%23').replace('?', '%3F').replace('&', '%26').replace('=', '%3D')
                    # path = result.url.replace('https://', '').replace('http://', '')


                    # 创建条目并添加
                    item = Item()
                    item.get_path = lambda: path
                    item.get_contentprovider = lambda: StringProvider(result.content)
                    item.get_title = lambda: result.url
                    item.get_mimetype = lambda: result.mimetype
                    item.get_hints = lambda: {Hint.COMPRESS: True}
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
                    index_content_provider = StringProvider(index_content)
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


async def main():
    """主函数"""
    #searching_domain = "scp-wiki-cn.wikidot.com/cytus-3"
    searching_domain = "scp-wiki-cn.wikidot.com/scp-cn-3959"
    
    crawler = WebCrawler(
        searching_domain=searching_domain.replace("https://", "").replace("http://", ""),
        max_retries=3,
        waiting_seconds_if_error=3.0,
        max_concurrent=10,
        store_response_text=True,
        fetch_resources=True,
        store_resource_content=True,
    )

    results = await crawler.crawl()

    # 尝试保存为ZIM文件
    zim_filename = f"{searching_domain.replace(".", "_").replace(r'/', '_')}-{time.strftime('%Y-%m-%d_%H-%M-%S')}.zim"
    crawler.save_to_zim(
        output_file=zim_filename,
        title=f"{searching_domain} 爬虫存档",
        description=f"{searching_domain} 网站的离线存档, 由SCP_Spider创建",
        creator_meta="SCP_Spider",
    )


if __name__ == "__main__":
    asyncio.run(main())
