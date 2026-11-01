import { test, expect } from "@playwright/test";
import { mockApi, GAP_ID } from "./fixtures";

// "Show on map" from a gap's detail screen -> the Map screen loads with both
// sides of that gap highlighted, then the embedding view + the
// topic-vs-topic similarity comparison both work end to end.
test("show on map from a gap, then compare two topics' similarity", async ({ page }) => {
  await mockApi(page);

  await page.goto(`/fields/default/gaps/${GAP_ID}`);
  await expect(page.getByRole("heading", { name: /barely cites/ })).toBeVisible();

  await page.getByRole("link", { name: "Show on map →" }).click();
  await expect(page).toHaveURL(/\/fields\/default\/map\?a=0&b=1/);
  await expect(page.getByRole("heading", { name: "Map" })).toBeVisible();
  await expect(page.getByText(/Highlighting the gap between/)).toBeVisible();
  await expect(page.locator("canvas")).toBeVisible();

  await page.getByRole("button", { name: "Embedding map" }).click();
  await expect(page.getByRole("button", { name: "Embedding map" })).toHaveAttribute("aria-pressed", "true");

  // The highlight pre-selects both compare dropdowns to the gap's two topics.
  await page.getByRole("button", { name: "Compare" }).click();
  await expect(page.getByText(/near-identical in topic/)).toBeVisible();
});
