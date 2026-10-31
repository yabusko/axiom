import { test, expect } from "@playwright/test";
import { mockApi } from "./fixtures";

// The task's own happy-path scenario: start -> open a gap -> draft a
// direction -> keep it -> export. API responses are mocked (see fixtures.ts)
// so this runs fast/free/deterministically, without a live backend or LLM
// spend -- docs/product/PRODUCT.md's confirmed demo scenario is "start from
// an already-built field," which is exactly what's mocked here.
test("start -> open a gap -> draft a direction -> keep it -> export", async ({ page }) => {
  await mockApi(page);

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Start a field" })).toBeVisible();
  await page.getByRole("link", { name: "retrieval-augmented generation" }).click();

  await expect(page).toHaveURL(/\/fields\/default\/landscape/);
  await expect(page.getByRole("heading", { name: "Landscape" })).toBeVisible();
  await expect(page.getByText("300", { exact: true })).toBeVisible();

  await page.getByRole("link", { name: "Research gaps" }).click();
  await expect(page).toHaveURL(/\/fields\/default\/gaps$/);
  await page.getByText(/barely cites/).click();

  await expect(page).toHaveURL(/\/fields\/default\/gaps\/0/);
  await expect(page.getByRole("heading", { name: /barely cites/ })).toBeVisible();
  await page.getByRole("button", { name: "Draft direction" }).click();

  await expect(page).toHaveURL(/\/fields\/default\/shortlist/);
  await expect(page.getByRole("heading", { name: "Draft" })).toBeVisible();
  await expect(page.getByText("Bridging NLP methods and QA-focused work")).toBeVisible();

  await page.getByRole("button", { name: "Keep" }).click();
  await expect(page.getByRole("heading", { name: "Kept" })).toBeVisible();

  await page.getByRole("button", { name: "Export to Markdown" }).click();
  await expect(page.getByText(/retrieval-augmented generation — shortlist/)).toBeVisible();
});

test("paper panel opens with evidence-span highlighting", async ({ page }) => {
  await mockApi(page);
  await page.goto("/fields/default/gaps/0");
  await page.getByRole("button", { name: "Semantic Parsing on Freebase from Question-Answer Pairs" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Unverified")).toBeVisible();
  await expect(page.locator(".abstract .ev-method")).toContainText("semantic parser");
});
