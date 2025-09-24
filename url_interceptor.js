// url_interceptor.js
(function() {
    'use strict';

    // 这个 URL_MAP 会被你的 Python 脚本注入
    const URL_MAP = {}; 

    // 获取当前 ZIM 文件的基本路径，这在 Kiwix-JS PWA 中至关重要
    // 例如：/www/content/scp-wiki-cn.wikidot.com_2023-10-27_10-30-00.zim/A
    // 我们需要找到 ZIM 内容的根，以便正确解析相对路径
    function getZimContentRoot() {
        const scripts = document.getElementsByTagName('script');
        // 找到我们自己这个脚本的 src
        const thisScriptSrc = Array.from(scripts).find(s => s.src.endsWith('/url_interceptor.js'))?.src;
        if (thisScriptSrc) {
            // 从脚本的 URL 中推断出 ZIM 内容的根路径
            // e.g., "https://pwa.kiwix.org/.../A/url_interceptor.js" -> "https://pwa.kiwix.org/.../A"
            return thisScriptSrc.substring(0, thisScriptSrc.lastIndexOf('/'));
        }
        // 如果找不到，提供一个回退（虽然不太可能发生）
        return '.';
    }
    const ZIM_ROOT = getZimContentRoot();


    function resolveUrl(requestedUrl) {
        // 首先，尝试将请求的 URL 标准化为绝对 URL
        // 注意：new URL(path, base) 是处理相对路径的健壮方法
        const absoluteUrl = new URL(requestedUrl, window.location.href).href;

        // 在我们的映射表中查找这个绝对 URL
        const zimPath = URL_MAP[absoluteUrl];
        
        if (zimPath) {
            // 如果找到，构建在 Kiwix-JS PWA 中有效的完整路径
            // zimPath 是像 "/domain.com/path/to/resource.css"
            // 我们需要把它变成 ZIM_ROOT + zimPath
            const finalUrl = new URL(ZIM_ROOT + zimPath).href;
            console.log(`[Interceptor] Rewriting ${requestedUrl} to ${finalUrl}`);
            return finalUrl;
        }
        
        // 如果没找到，返回原始 URL
        return requestedUrl;
    }

    // --- Monkey-patch fetch ---
    const originalFetch = window.fetch;
    window.fetch = function(input, init) {
        let url = (typeof input === 'string') ? input : input.url;
        const rewrittenUrl = resolveUrl(url);
        
        if (typeof input === 'string') {
            return originalFetch(rewrittenUrl, init);
        } else {
            // 如果输入是 Request 对象，需要创建一个新的
            const newRequest = new Request(rewrittenUrl, input);
            return originalFetch(newRequest);
        }
    };

    // --- Monkey-patch XMLHttpRequest ---
    const originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, async, user, password) {
        const rewrittenUrl = resolveUrl(url);
        return originalOpen.call(this, method, rewrittenUrl, async, user, password);
    };

    console.log('URL Interceptor loaded and active.');

})();