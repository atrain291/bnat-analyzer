import { describe, it, expect } from "vitest";
import api from "../client";

describe("API client", () => {
  it("has correct base URL", () => {
    expect(api.defaults.baseURL).toBe("/api");
  });

  it("has JSON content type", () => {
    expect(api.defaults.headers["Content-Type"]).toBe("application/json");
  });
});
