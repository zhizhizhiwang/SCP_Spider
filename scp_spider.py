import asyncio
import os
from asyncio import sleep
import httpx  # 导入requests库，用于发送HTTP请求
from urllib.parse import (
    urljoin,
    urlparse,
)  # 从urllib.parse模块导入urljoin函数，用于拼接URL
from bs4 import BeautifulSoup  # 从bs4模块导入BeautifulSoup，用于解析HTML
import logging
import json  # 导入json库，用于处理JSON数据
from datetime import datetime  # 导入datetime模块，用于处理日期和时间

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(levelname)s] %(message)s"
)

waiting_seconds_if_error = 3
searching_domain = "tw.neuq.edu.cn"
max_retries = 3
# 设置需要跳过的文件扩展名
skipped_extensions = (
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".gif",
    ".jpeg",
    ".svg",
    ".ico",
    ".webp",
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
target_urls: set[str] = {f"https://{searching_domain}"}  # 初始化一个集合，包含目标URL
accessed_urls: set[str] = set()  # 初始化一个集合，用于存储已访问的URL
results: list[dict[str, str]] = []  # 初始化一个列表，用于存储结果
"""
[
    {
        "url": f"https://{searching_domain}",
        "success": True / False,
        "remark":"No HTTPS" / "bad domain",
        "status_code": 200,
        "text": "..."
    }
]
"""


def find_urls(url: str, response: str):
    """从HTML响应中提取所有链接"""
    global target_urls, accessed_urls  # 声明全局变量target_urls和accessed_urls

    # 使用BeautifulSoup解析HTML响应
    soup = BeautifulSoup(response, "html.parser")
    base_url = url  # 获取当前URL作为基准URL

    # 提取所有标签中的链接
    tags = ["a", "img", "script", "link"]  # 定义一个列表，包含需要提取链接的标签
    for tag in tags:
        for element in soup.find_all(tag):  # 遍历所有找到的标签
            attr = "href" if tag == "a" else "src"  # 根据标签类型确定属性名
            link = element.get(attr)  # 获取标签的属性值
            if link:
                full_url = urljoin(base_url, link)  # 拼接成完整的URL

                parsed_url = urlparse(full_url)  # 解析URL
                if parsed_url.netloc.endswith(
                    searching_domain
                ):  # 检查域名是否属于指定网站
                    target_urls.add(full_url)  # 将完整的URL添加到目标URL集合中


async def main():
    def __finish_request(url: str, response: httpx.Response, remark=""):
        """处理请求完成后的操作"""
        global accessed_urls, results
        accessed_urls.add(url)  # 将已访问的URL添加到集合中
        results.append(
            {
                "url": url,
                "success": response.is_success,
                "remark": remark if remark else response.reason_phrase,
                "status_code": response.status_code,
                "text": response.text,
            }
        )  # 将URL、响应文本和状态码添加到结果列表中

    global target_urls, accessed_urls
    async with httpx.AsyncClient() as client:  # 创建异步HTTP客户端
        while target_urls:
            url = target_urls.pop()  # 从集合中取出一个URL
            parsed_url = urlparse(url)  # 解析URL
            if os.path.splitext(parsed_url.path)[-1] in skipped_extensions:
                continue

            error = 0
            while True:
                try:
                    response = await client.get(
                        url, follow_redirects=True
                    )  # 发送GET请求获取响应
                except Exception as e:

                    error += 1

                    logging.error(
                        f"Error accessing {url}: {e}, waiting {waiting_seconds_if_error} seconds"
                    )
                    if error >= max_retries:
                        # 确认失败
                        if url.startswith("https://"):
                            # 如果HTTPS请求失败，降级到HTTP并重新添加到队列
                            target_urls.add(f"http://{url.removeprefix("https://")}")
                            break
                        else:
                            # 【最终出口1-失败】确认是链接原因，跳过
                            logging.error(f"Skipping {url} after {max_retries} retries")
                            __finish_request(url, response, "bad domain")  # 记录失败
                            break
                    else:
                        # 暂时失败，假定为频繁请求，等待一段时间再试
                        await sleep(waiting_seconds_if_error)  # 暂停，避免频繁请求
                        continue
                else:
                    break

            if error >= max_retries:
                continue

            # 【最终出口2-成功】
            __finish_request(
                url,
                response,
                "No HTTPS" if url.startswith("http://") else response.reason_phrase,
            )  # 记录成功
            try:
                find_urls(url, response)  # 调用find_urls函数提取链接并更新目标URL集合
            except Exception as e:
                logging.error(f"Error parsing {url}: {e}")
            target_urls -= accessed_urls  # 更新目标URL集合，去除已访问的URL
            # 不用明确显示已访问的URL，因为httpx会自动显示访问的URL
            logging.info(
                f"URLs: {100*len(accessed_urls)/(len(target_urls)+len(accessed_urls)):.2f} %, {len(accessed_urls)}/{len(target_urls)+len(accessed_urls)}"
            )
            # await sleep(0.5)

    logging.info("Finished crawling")  # 记录爬虫完成

    all_urls = target_urls.union(accessed_urls)  # 合并目标URL和已访问URL集合
    try:
        with open(
            f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f)
        logging.info("Results saved to results.json")  # 记录结果保存成功
    except Exception as e:

        def __csv_str_handle(s: str):
            s = s.replace("\n", "\\n").replace("\r", "\\r")
            if s.find(",") != -1:
                s = f'"{s}"'
            if s.find('"') != -1:
                s = f'"{s.replace('"', '""')}"'
            return s

        logging.error(f"Error saving results in JSON: {e}")
        with open(
            f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("URL,Success,Remark,Status_Code,Text\n")
            for url in all_urls:
                f.write(
                    f"{__csv_str_handle(url)},\
                        {__csv_str_handle(results[url]['success'])},\
                        {__csv_str_handle(results[url]['remark'])},\
                        {__csv_str_handle(results[url]['status_code'])},\
                        {__csv_str_handle(results[url]['text'])}\n"
                )
        logging.info("Results saved to results.csv")


if __name__ == "__main__":
    asyncio.run(main())
