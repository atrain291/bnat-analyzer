import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ProcessingStatus from "./pages/ProcessingStatus";
import VideoReview from "./pages/VideoReview";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/processing/:performanceId" element={<ProcessingStatus />} />
          <Route path="/review/:performanceId" element={<VideoReview />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
