import { describe, it, expect, vi, beforeEach } from "vitest";
import api from "../client";
import { getPerformanceStatus, getPerformance, deletePerformance } from "../performances";

vi.mock("../client", () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
    defaults: { baseURL: "/api", headers: { "Content-Type": "application/json" } },
  },
}));

describe("performances API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("getPerformanceStatus fetches status", async () => {
    const mockStatus = { id: 1, status: "processing", pipeline_progress: { stage: "pose" }, error: null };
    vi.mocked(api.get).mockResolvedValue({ data: mockStatus });

    const result = await getPerformanceStatus(1);
    expect(api.get).toHaveBeenCalledWith("/performances/1/status");
    expect(result).toEqual(mockStatus);
  });

  it("getPerformance fetches full performance", async () => {
    const mockPerf = { id: 1, status: "complete", frames: [], analysis: [] };
    vi.mocked(api.get).mockResolvedValue({ data: mockPerf });

    const result = await getPerformance(1);
    expect(api.get).toHaveBeenCalledWith("/performances/1");
    expect(result).toEqual(mockPerf);
  });

  it("deletePerformance calls delete", async () => {
    vi.mocked(api.delete).mockResolvedValue({ data: {} });

    await deletePerformance(1);
    expect(api.delete).toHaveBeenCalledWith("/performances/1");
  });
});
