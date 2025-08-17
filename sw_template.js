// sw_template.js

// 这个占位符将在Python中被替换为真实的URL映射
const URL_MAP = {}; 

const CACHE_NAME = 'offline-archive-cache-v1';

// 安装Service Worker时，可以预缓存一些核心资源（可选）
self.addEventListener('install', (event) => {
    console.log('[SW] Service Worker installed.');
    event.waitUntil(self.skipWaiting()); // 强制新的SW立即激活
});

self.addEventListener('activate', (event) => {
    console.log('[SW] Service Worker activated.');
    // 清理旧缓存（如果需要）
    event.waitUntil(self.clients.claim()); // 控制所有打开的页面
});

// --- 核心：拦截网络请求 ---
self.addEventListener('fetch', (event) => {
    const request = event.request;
    const requestUrl = new URL(request.url);

    // 忽略非http/https请求
    if (!requestUrl.protocol.startsWith('http')) {
        return;
    }
    
    console.log('[SW] Intercepting fetch for:', request.url);

    // 优先从我们的自定义映射中查找
    // URL_MAP 的键是原始的绝对URL，值是ZIM内部的相对路径
    if (URL_MAP[request.url]) {
        const localPath = URL_MAP[request.url];
        console.log(`[SW] Remapping ${request.url} to local ZIM path: ${localPath}`);
        // 使用ZIM内部路径发起新的fetch请求
        // 注意：在Kiwix环境中，fetch可以直接处理指向ZIM内部的相对路径
        event.respondWith(fetch(localPath));
        return;
    }

    // 对于在JS中动态生成的、但我们没有在MAP中定义的绝对路径
    // 我们可以尝试根据其路径在ZIM中猜测一个位置
    // 例如：请求 https://example.com/api/data，我们尝试访问 /example.com/api/data
    const zimPathGuess = `/${requestUrl.hostname}${requestUrl.pathname}${requestUrl.search}`;
    console.log(`[SW] No exact map match. Guessing ZIM path: ${zimPathGuess}`);
    event.respondWith(
        fetch(zimPathGuess).catch(err => {
            console.error(`[SW] Fetch failed for both map and guess (${request.url}):`, err);
            // 如果所有尝试都失败，返回一个网络错误
            return new Response('Network error trying to fetch from ZIM archive', {
                status: 404,
                statusText: 'Not Found in ZIM'
            });
        })
    );
});