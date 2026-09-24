import { spawn } from "child_process";
import http from "http";
import WebSocket from "ws";

const targetUrl = process.argv[2] || "http://127.0.0.1:8080/";
const port = 9222;
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

console.log(`[TestBrowser] Testing URL: ${targetUrl}`);

// 1. Launch Chrome headless with remote debugging
const chromeProc = spawn(chromePath, [
  "--headless=new",
  `--remote-debugging-port=${port}`,
  "--remote-allow-origins=*",
  "--disable-gpu",
  "--no-sandbox",
  "--user-data-dir=/tmp/chrome-test-profile-" + Date.now(),
  "about:blank",
]);

let killed = false;
function cleanup() {
  if (!killed) {
    killed = true;
    try {
      chromeProc.kill("SIGTERM");
    } catch {}
  }
}
process.on("exit", cleanup);
process.on("SIGINT", () => {
  cleanup();
  process.exit(1);
});

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function getWsUrl() {
  for (let i = 0; i < 30; i++) {
    await sleep(200);
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      const data = await res.json();
      if (data && data.webSocketDebuggerUrl) {
        return data.webSocketDebuggerUrl;
      }
    } catch {}
  }
  throw new Error("Failed to connect to Chrome remote debugging port.");
}

async function run() {
  try {
    const wsUrl = await getWsUrl();
    const ws = new WebSocket(wsUrl);

    let id = 1;
    const callbacks = new Map();

    function send(method, params = {}) {
      return new Promise((resolve, reject) => {
        const msgId = id++;
        callbacks.set(msgId, { resolve, reject });
        ws.send(JSON.stringify({ id: msgId, method, params }));
      });
    }

    await new Promise((resolve) => ws.on("open", resolve));

    // Enable events
    await send("Target.setDiscoverTargets", { discover: true });
    
    // Create new target page
    const target = await send("Target.createTarget", { url: targetUrl });
    const targetId = target.targetId;

    // Attach to target
    const attached = await send("Target.attachToTarget", {
      targetId,
      flatten: true,
    });
    const sessionId = attached.sessionId;

    function sendSession(method, params = {}) {
      return new Promise((resolve, reject) => {
        const msgId = id++;
        callbacks.set(msgId, { resolve, reject });
        ws.send(JSON.stringify({ id: msgId, sessionId, method, params }));
      });
    }

    const consoleLogs = [];
    const networkRequests = [];

    ws.on("message", (raw) => {
      const data = JSON.parse(raw);
      if (data.id && callbacks.has(data.id)) {
        const cb = callbacks.get(data.id);
        callbacks.delete(data.id);
        if (data.error) cb.reject(data.error);
        else cb.resolve(data.result);
        return;
      }

      // Handle events
      if (data.method === "Runtime.consoleAPICalled") {
        const type = data.params.type;
        const text = (data.params.args || [])
          .map((a) => (a.value !== undefined ? a.value : JSON.stringify(a)))
          .join(" ");
        consoleLogs.push(`[CONSOLE ${type.toUpperCase()}] ${text}`);
      } else if (data.method === "Runtime.exceptionThrown") {
        const desc = data.params.exceptionDetails?.exception?.description ||
          data.params.exceptionDetails?.text;
        consoleLogs.push(`[UNCAUGHT EXCEPTION] ${desc}`);
      } else if (data.method === "Network.responseReceived") {
        const r = data.params.response;
        networkRequests.push({
          url: r.url,
          status: r.status,
          mimeType: r.mimeType,
          headers: r.headers,
        });
      }
    });

    await sendSession("Runtime.enable");
    await sendSession("Network.enable");
    await sendSession("Page.enable");

    console.log("[TestBrowser] Waiting for page load and runtime execution...");
    // Wait for model initialization and first inference (approx 5 seconds)
    await sleep(6000);

    // Evaluate page state
    const evalRes = await sendSession("Runtime.evaluate", {
      expression: `
        JSON.stringify({
          crossOriginIsolated: window.crossOriginIsolated,
          hasSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
          hasWebGPU: 'gpu' in navigator,
          statusText: document.getElementById('status-text')?.textContent,
          bannerText: document.getElementById('alert-banner')?.textContent,
          bannerVisible: document.getElementById('alert-banner')?.style.display !== 'none',
          engine: document.getElementById('metric-provider')?.textContent,
          latency: document.getElementById('metric-latency')?.textContent,
          throughput: document.getElementById('metric-fps')?.textContent,
          topStep: document.getElementById('metric-top-step')?.textContent,
          canvasWidth: document.getElementById('main-canvas')?.width,
          canvasHeight: document.getElementById('main-canvas')?.height,
        })
      `,
      returnByValue: true,
    });

    const pageState = JSON.parse(evalRes.result.value);

    console.log("\n=== BROWSER DIAGNOSTIC REPORT ===");
    console.log(`URL: ${targetUrl}`);
    console.log(`crossOriginIsolated: ${pageState.crossOriginIsolated}`);
    console.log(`hasSharedArrayBuffer: ${pageState.hasSharedArrayBuffer}`);
    console.log(`hasWebGPU: ${pageState.hasWebGPU}`);
    console.log(`Status Text: ${pageState.statusText}`);
    console.log(`Alert Banner Visible: ${pageState.bannerVisible}`);
    if (pageState.bannerVisible) {
      console.log(`Alert Banner Text: ${pageState.bannerText}`);
    }
    console.log(`Engine: ${pageState.engine}`);
    console.log(`Latency: ${pageState.latency}`);
    console.log(`Throughput: ${pageState.throughput}`);
    console.log(`Top Step: ${pageState.topStep}`);
    console.log(`Canvas: ${pageState.canvasWidth}x${pageState.canvasHeight}`);

    console.log("\n=== NETWORK REQUESTS ===");
    for (const req of networkRequests) {
      const shortUrl = req.url.length > 80 ? req.url.slice(0, 77) + "..." : req.url;
      console.log(`${req.status} | ${req.mimeType} | ${shortUrl}`);
    }

    console.log("\n=== CONSOLE LOGS ===");
    for (const log of consoleLogs) {
      console.log(log);
    }

    cleanup();
    process.exit(0);
  } catch (err) {
    console.error("[TestBrowser] Error:", err);
    cleanup();
    process.exit(1);
  }
}

run();
