// src/App.jsx
import React from "react";
import { HashRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";

import HomePage from "./pages/HomePage";
import UploadPage from "./pages/UploadPage";
import WebcamPage from "./pages/WebcamPage";
import AboutPage from "./pages/AboutPage";

function App() {
  return (
    <Router>
      {/* Navbar shown on every page */}
      <Navbar />

      {/* Routes */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/webcam" element={<WebcamPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Router>
  );
}

export default App;
