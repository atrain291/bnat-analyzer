import api from "./client";

export interface Dancer {
  id: number;
  name: string;
  experience_level: string | null;
  created_at: string;
}

export async function createDancer(name: string, experienceLevel?: string): Promise<Dancer> {
  const { data } = await api.post("/dancers/", {
    name,
    experience_level: experienceLevel || null,
  });
  return data;
}

export async function listDancers(): Promise<Dancer[]> {
  const { data } = await api.get("/dancers/");
  return data;
}
