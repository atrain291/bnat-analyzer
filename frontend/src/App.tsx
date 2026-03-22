import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ProcessingStatus from "./pages/ProcessingStatus";
import DancerSelection from "./pages/DancerSelection";
import VideoReview from "./pages/VideoReview";
import MultiAngleUpload from "./pages/MultiAngleUpload";
import MultiAngleProcessing from "./pages/MultiAngleProcessing";
import MultiAngleDancerLink from "./pages/MultiAngleDancerLink";
import MultiAngleReview from "./pages/MultiAngleReview";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/processing/:performanceId" element={<ProcessingStatus />} />
          <Route path="/select-dancers/:performanceId" element={<DancerSelection />} />
          <Route path="/review/:performanceId" element={<VideoReview />} />
          <Route path="/multi-angle/upload" element={<MultiAngleUpload />} />
          <Route path="/multi-angle/processing/:groupId" element={<MultiAngleProcessing />} />
          <Route path="/multi-angle/link-dancers/:groupId" element={<MultiAngleDancerLink />} />
          <Route path="/multi-angle/review/:groupId" element={<MultiAngleReview />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
