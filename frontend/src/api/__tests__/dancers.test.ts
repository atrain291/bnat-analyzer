import { describe, it, expect, vi, beforeEach } from "vitest";
import api from "../client";
import { createDancer, listDancers } from "../dancers";

vi.mock("../client", () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    defaults: { baseURL: "/api", headers: { "Content-Type": "application/json" } },
  },
}));

describe("dancers API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("createDancer posts to /dancers/", async () => {
    const mockDancer = { id: 1, name: "Meera", experience_level: "beginner", created_at: "2024-01-01" };
    vi.mocked(api.post).mockResolvedValue({ data: mockDancer });

    const result = await createDancer("Meera", "beginner");
    expect(api.post).toHaveBeenCalledWith("/dancers/", { name: "Meera", experience_level: "beginner" });
    expect(result).toEqual(mockDancer);
  });

  it("listDancers gets from /dancers/", async () => {
    const mockDancers = [{ id: 1, name: "Meera" }];
    vi.mocked(api.get).mockResolvedValue({ data: mockDancers });

    const result = await listDancers();
    expect(api.get).toHaveBeenCalledWith("/dancers/");
    expect(result).toEqual(mockDancers);
  });
});
