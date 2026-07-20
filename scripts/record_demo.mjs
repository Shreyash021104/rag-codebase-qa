// Records a screen capture of the RAG Q&A web UI in action against the local
// server. Uses an already-indexed repo so there's no dead indexing wait on
// screen. Run from the collab-editor client dir (which has Playwright):
//   node <this> [baseUrl] [outDir]
import { chromium } from "playwright";
import path from "node:path";

const baseUrl = process.argv[2] ?? "http://localhost:8000";
const outDir = process.argv[3] ?? "/Users/shreyashpatange/letsbuild/notes/demo-video-3/raw";
const VIEWPORT = { width: 1280, height: 820 };

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: VIEWPORT,
  recordVideo: { dir: outDir, size: VIEWPORT },
});
const page = await ctx.newPage();

async function typeInto(sel, text, delay = 40) {
  await page.click(sel);
  await page.type(sel, text, { delay });
}

await page.goto(baseUrl);
await page.waitForSelector(".repo");
await page.waitForTimeout(1200);

// Select the indexed rate-limiter repo (the one with 92 chunks).
const repo = page.locator(".repo", { hasText: "distributed-rate-limiter-gateway" });
await repo.click();
await page.waitForSelector("#ask-panel:not([hidden])");
await page.waitForTimeout(800);

// --- Question 1: a semantic question ---
await typeInto("#question", "what happens if redis is unreachable when checking a rate limit?");
await page.waitForTimeout(400);
await page.click("#ask-btn");
// Wait for the synthesized answer paragraph to appear.
await page.waitForSelector(".answer", { timeout: 40000 });
await page.waitForTimeout(1800);

// Expand the first cited source to reveal the real code.
await page.locator("details.source summary").first().click();
await page.waitForTimeout(2200);
// Collapse it again before the next question.
await page.locator("details.source summary").first().click();
await page.waitForTimeout(600);

// --- Question 2: an exact-identifier question ---
await page.fill("#question", "");
await typeInto("#question", "where is the token bucket algorithm implemented?");
await page.waitForTimeout(400);
await page.click("#ask-btn");
await page.waitForSelector(".answer", { timeout: 40000 });
await page.waitForTimeout(2000);
await page.locator("details.source summary").first().click();
await page.waitForTimeout(2600);

const videoPath = await page.video().path();
await browser.close();
console.log("recorded:", videoPath);
