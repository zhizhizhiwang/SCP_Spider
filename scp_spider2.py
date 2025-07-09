import asyncio
import os
import mimetypes
from asyncio import sleep
import httpx
from urllib.parse import urljoin, urlparse
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
    extensions = (
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".gif",
        ".jpeg",
        ".svg",
        ".ico",
        ".webp",
    )
    return any(url.lower().endswith(ext) for ext in extensions)


@dataclass
class CrawlResult:
    url: str
    success: bool
    remark: str
    status_code: int
    text: str = ""
    content: bytes | str = field(default=b"", repr=False)
    mimetype: str = "text/html"
    is_resource: bool = False


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

    def find_urls(self, url: str, response_text: str) -> tuple[set[str], set[str]]:
        """从HTML响应中提取所有链接和资源

        返回：
            (页面链接, 资源链接)
        """
        new_urls = set()
        resource_urls = set()
        try:
            soup = BeautifulSoup(response_text, "html.parser")
            base_url = url

            tags_attrs = [
                ("a", "href"),  # 页面链接
                ("img", "src"),  # 图片
                ("script", "src"),  # JS
                ("link", "href"),  # CSS 
                ("source", "src"),  # 媒体元素
                ("video", "src"),  # 视频
                ("audio", "src"),  # 音频
            ]

            for tag, attr in tags_attrs:
                for element in soup.find_all(tag):
                    link = element.get(attr)
                    if not link:
                        continue

                    full_url = urljoin(base_url, link)
                    parsed_url = urlparse(full_url)

                    # 处理内部链接
                    # if parsed_url.netloc.startswith(urlparse((f'https://{self.searching_domain}')).netloc):
                    if(parsed_url.netloc == self.start_domain.netloc and parsed_url.path.startswith(self.start_domain.path)):
                        if tag == "a":
                            # 页面链接
                            new_urls.add(full_url)
                        elif self.fetch_resources and is_resource_file(full_url):
                            # 内部资源
                            resource_urls.add(full_url)

                    # 处理外部资源（如果启用）
                    elif (
                        self.fetch_resources
                        and tag != "a"
                        and is_resource_file(full_url)
                    ):
                        # 外部资源（非链接）
                        resource_urls.add(full_url)
            for element in soup.find_all("style"): # <style> url('') </style> 处理
                regex = r"url\(['\"]([^'\"]+)['\"]\)"
                for match in re.findall(regex, str(element)):
                    resource_urls.add(match) if urlparse(match).netloc != '' else resource_urls.add(urljoin(base_url, match))
        except Exception as e:
            logging.error(f"Error parsing URLs from {url}: {e}")

        return new_urls, resource_urls

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
                        url=url,
                        success=response.is_success,
                        remark=remark,
                        status_code=response.status_code,
                        text="",  # 资源文件不保存文本
                        content=(
                            response.content if self.store_resource_content else b""
                        ),
                        mimetype=mimetype,
                        is_resource=True,
                    )
                else:
                    # HTML页面保存文本内容
                    text = response.text if self.store_response_text else ""
                    result = CrawlResult(
                        url=url,
                        success=response.is_success,
                        remark=remark,
                        status_code=response.status_code,
                        text=text,
                        content=response.content,  # 同时保存二进制内容用于ZIM文件
                        mimetype=mimetype,
                        is_resource=False,
                    )

                    # 提取新URL和资源URL
                    new_urls, new_resource_urls = self.find_urls(url, response.text)

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

    def save_results(self, results: list[dict[str, Any]] = None) -> None:
        """保存结果到JSON文件"""
        if results is None:
            results = [asdict(result) for result in self.results]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 移除二进制内容以便JSON序列化
        json_results = []
        for result in results:
            result_copy = result.copy()
            # 移除二进制内容字段
            if "content" in result_copy:
                result_copy["content"] = (
                    f"<{len(result_copy.get('content', b''))} bytes>"
                )
            json_results.append(result_copy)

        # 尝试保存为JSON
        try:
            filename = f"results_{timestamp}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving results as JSON: {e}")
            self._save_as_csv(json_results, timestamp)

    def _save_as_csv(self, results: list[dict[str, Any]], timestamp: str) -> None:
        """保存结果为CSV格式"""
        try:
            filename = f"results_{timestamp}.csv"
            with open(filename, "w", encoding="utf-8", newline="") as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving results as CSV: {e}")

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
                    logging.info(f'生成路径：{path}')

                    # 创建条目并添加
                    content_provider = StringProvider(result.content)
                    item = Item()
                    item.get_path = lambda: path
                    item.get_title = lambda: result.url
                    item.get_mimetype = lambda: result.mimetype
                    item.get_contentprovider = lambda: content_provider
                    item.get_hints = lambda: {Hint.COMPRESS: True}
                    creator.add_item(item)  # 在上下文中添加

                    # 更新索引
                    if result.is_resource:
                        file_count += 1
                        index_content += f'<li><a href="{path}">{result.url}</a></li>'
                    else:
                        page_count += 1
                        index_content += f'<li><a href="{path}">{result.url}</a></li>'

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
            return False


async def main():
    """主函数"""
    searching_domain = "scp-wiki-cn.wikidot.com/cytus-3"

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
