import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mockApi } from "./fixtures";

// 0 serious/critical violations, per the task's quality floor.
const SCREENS = [
  { name: "start", path: "/" },
  { name: "landscape", path: "/fields/default/landscape" },
  { name: "gap-detail", path: "/fields/default/gaps/0" },
  { name: "shortlist", path: "/fields/default/shortlist" },
];

for (const { name, path } of SCREENS) {
  test(`axe: ${name} has no serious/critical violations`, async ({ page }) => {
    await mockApi(page);
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });
}
