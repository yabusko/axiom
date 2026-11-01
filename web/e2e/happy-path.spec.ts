import { test, expect } from "@playwright/test";
import { mockApi, GAP_ID } from "./fixtures";

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
  await expect(page.getByText("220", { exact: true })).toBeVisible();

  await page.getByRole("link", { name: "Research gaps" }).click();
  await expect(page).toHaveURL(/\/fields\/default\/gaps$/);
  await page.getByText(/shares 7 citations with/).click();

  await expect(page).toHaveURL(new RegExp(`/fields/default/gaps/${GAP_ID}`));
  await expect(page.getByRole("heading", { name: /shares 7 citations with/ })).toBeVisible();

  await page.getByRole("button", { name: "Check against OpenAlex" }).click();
  await expect(page.getByText(/12 OpenAlex works combine both sides/)).toBeVisible();
  await expect(page.getByText(/genuinely under-explored pairing/)).toBeVisible();

  await page.getByRole("button", { name: "Draft direction" }).click();

  await expect(page).toHaveURL(/\/fields\/default\/shortlist/);
  await expect(page.getByRole("heading", { name: "Draft" })).toBeVisible();
  await expect(page.getByText("Bridging retrieve-and-read methods and re-ranking-based open-domain QA")).toBeVisible();

  await page.getByRole("button", { name: "Keep" }).click();
  await expect(page.getByRole("heading", { name: "Kept" })).toBeVisible();

  await page.getByRole("button", { name: "Export to Markdown" }).click();
  await expect(page.getByText(/retrieval-augmented generation — shortlist/)).toBeVisible();
});

test("instant field creation redirects straight to landscape, no polling", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Start a field" })).toBeVisible();

  await page.getByLabel("Topic").fill("retrieval augmented generation");
  await page.getByRole("button", { name: "Find this topic" }).click();

  // The mocked POST /fields returns status "ready" synchronously -- no
  // build-progress screen, straight to the landscape.
  await expect(page).toHaveURL(/\/fields\/default\/landscape/);
  await expect(page.getByRole("heading", { name: "Landscape" })).toBeVisible();
});

test("paper panel opens with evidence-span highlighting", async ({ page }) => {
  await mockApi(page);
  await page.goto(`/fields/default/gaps/${GAP_ID}`);
  await page.getByRole("button", { name: "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Unverified")).toBeVisible();
  // Real data quirk: the "open domain question answering" (task) and
  // "Generative model" (method) entities share the exact same evidence_span
  // text -- highlightAbstract() keeps only the first (higher-confidence)
  // match for an overlapping span, so only .ev-task renders here, not
  // .ev-method. .ev-dataset ("Natural Questions...") has its own span.
  await expect(page.locator(".abstract .ev-task")).toContainText("open domain question answering");
  await expect(page.locator(".abstract .ev-dataset")).toContainText("Natural Questions");
});
