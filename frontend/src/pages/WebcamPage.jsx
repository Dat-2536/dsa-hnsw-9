// src/pages/WebcamPage.jsx
import React, { useEffect, useRef, useState } from "react";
import { recognizeFrame } from "../services/api";
import FaceBox from "../components/FaceBox";

// Hàm helper: convert distance -> similarity %
function distanceToSimilarity(distance, maxDistance = 0.5){
  return sigmoid_similarity(distance, maxDistance, 15);
}

function sigmoid_similarity(distance, threshold=0.5, alpha=20) {
  if (typeof distance !== "number" || Number.isNaN(distance)) {
    return 0;
  }
  const expComponent = Math.exp(alpha * (distance - threshold));
  const similarity = 100 / (1 + expComponent);
  return Math.round(similarity);
}

const WebcamPage = () => {
  const videoRef = useRef(null);
  const overlayRef = useRef(null); // Canvas phủ lên để vẽ khung
  const [results, setResults] = useState([]);
  const [running, setRunning] = useState(true);

  // 1. Khởi động Webcam
  useEffect(() => {
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            // Không set cứng width/height -> để trình duyệt tự chọn tốt nhất cho thiết bị
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error(err);
        alert("Không truy cập được webcam. Hãy kiểm tra quyền truy cập.");
      }
    }

    initCamera();

    // Cleanup
    return () => {
      const video = videoRef.current;
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // 2. Vòng lặp gửi API (Xử lý ngầm, không ảnh hưởng hiển thị)
  useEffect(() => {
    if (!running) return;

    const interval = setInterval(captureAndSendFrame, 800);
    return () => clearInterval(interval);

    async function captureAndSendFrame() {
      const video = videoRef.current;
      // Chỉ chụp khi video đã sẵn sàng và có kích thước
      if (!video || video.readyState !== 4 || video.videoWidth === 0) return;

      // Tạo canvas ảo để chụp frame gửi đi
      const offscreenCanvas = document.createElement("canvas");
      offscreenCanvas.width = video.videoWidth;
      offscreenCanvas.height = video.videoHeight;
      const ctx = offscreenCanvas.getContext("2d");

      // Vẽ frame gốc vào canvas ảo
      ctx.drawImage(video, 0, 0);

      // Nén ảnh JPEG 0.7
      const base64Image = offscreenCanvas.toDataURL("image/jpeg", 0.7);

      try {
        const res = await recognizeFrame(base64Image);

        // Chuẩn hoá dữ liệu trả về từ API
        let rawFaces = [];
        if (Array.isArray(res?.faces)) {
          rawFaces = res.faces;
        } else if (res && (res.box || res.info || res.status)) {
          rawFaces = [res];
        }

        const processedFaces = rawFaces.map((face) => {
            const info = face.info || {};
            // Ưu tiên ảnh crop từ backend trả về để đỡ tốn sức frontend
            const imgSrc = face.crop_image || face.imgSrc || "https://placehold.co/100x100?text=No+Image";

            return {
              ...face,
              mssv: face.student_id || info.MSSV || "Unknown",
              name: face.name || info.Ten || "Unknown",
              distance: face.distance || 0,
              imgSrc: imgSrc,
              box: face.box // [top, right, bottom, left]
            };
        });

        setResults(processedFaces);
      } catch (err) {
        console.error("API Error:", err);
      }
    }
  }, [running]);

  // 3. Vẽ khung xanh (Overlay) mỗi khi có kết quả mới
  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;

    if (!canvas || !video || video.videoWidth === 0) return;

    // --- QUAN TRỌNG: Đồng bộ kích thước Canvas phủ khớp với Video ---
    // Video hiển thị bao nhiêu thì Canvas phủ bấy nhiêu
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Xóa khung cũ

    if (results.length === 0) return;

    // Cấu hình style vẽ
    ctx.lineWidth = 3;
    ctx.font = "bold 18px Segoe UI, sans-serif";
    ctx.textBaseline = "top";

    results.forEach((face) => {
      if (!Array.isArray(face.box)) return;
      const [top, right, bottom, left] = face.box;
      const width = right - left;
      const height = bottom - top;

      // Màu sắc: Đỏ nếu Unknown, Xanh nếu nhận diện được
      const isUnknown = face.name === "Unknown";
      const color = isUnknown ? "#dc3545" : "#0d6efd";

      // 1. Vẽ khung
      ctx.strokeStyle = color;
      ctx.strokeRect(left, top, width, height);

      // 2. Vẽ nền chữ
      const similarity = distanceToSimilarity(face.distance);
      const label = `${face.name} (${similarity}%)`;
      const textWidth = ctx.measureText(label).width + 12;
      const textHeight = 28;

      ctx.fillStyle = color;
      ctx.fillRect(left, top - textHeight, textWidth, textHeight);

      // 3. Vẽ chữ (Màu trắng, không bị ngược vì đã xóa scaleX)
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, left + 6, top - textHeight + 4);
    });

  }, [results]); // Vẽ lại khi data thay đổi

  return (
    <div className="d-flex flex-column min-vh-90 bg-dark text-light">
      <main className="flex-grow-1 py-3">
        <div className="container">
          <div className="mb-4">
            <h2 className="h4 mb-1 text-light">Chế độ Camera trực tiếp</h2>
            <p className="small text-light mb-0" style={{ opacity: 0.8 }}>
              Hệ thống sẽ chụp khung hình định kỳ từ webcam và gửi lên backend
              để nhận diện khuôn mặt theo thời gian thực.
            </p>
          </div>

          <div className="row g-4">
            {/* Khung webcam – col-lg-8 */}
            <div className="col-lg-8">
              <div className="card bg-dark border-secondary">
                <div className="card-body p-2">
                  <h5 className="card-title mb-3 text-light px-2 pt-2">Live Camera</h5>

                  {/* WRAPPER QUAN TRỌNG:
                      - position-relative: Để canvas nằm đè lên video
                      - w-100: Chiếm hết chiều ngang cột
                      - lineHeight: 0 để xóa khoảng trắng thừa dưới video
                   */}
                  <div className="position-relative w-100 bg-black rounded overflow-hidden" style={{ lineHeight: 0 }}>
                    
                    {/* LAYER 1: VIDEO (Hiển thị mượt) */}
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted // Mute để tránh lỗi auto-play policy
                      style={{
                        width: "100%",      // Co giãn theo khung cha
                        height: "auto",     // Tự động theo tỷ lệ aspect ratio
                        display: "block",
                        // KHÔNG CÓ transform: scaleX(-1) -> Hiển thị đúng chiều
                      }}
                    />

                    {/* LAYER 2: CANVAS (Vẽ khung, Trong suốt) */}
                    <canvas
                      ref={overlayRef}
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",      // Phủ kín video
                        height: "100%",
                        pointerEvents: "none", // Để chuột click xuyên qua được (nếu cần)
                      }}
                    />
                  </div>

                  {/* Nút điều khiển */}
                  <div className="d-flex gap-2 mt-3 px-2 pb-2 align-items-center">
                    <button
                      type="button"
                      className={`btn btn-sm rounded-pill fw-semibold ${
                        running ? "btn-outline-warning" : "btn-success"
                      }`}
                      onClick={() => setRunning((prev) => !prev)}
                    >
                      {running ? "Tạm dừng gửi frame" : "Tiếp tục gửi frame"}
                    </button>
                    
                    <p className="small mb-0 text-light ms-auto" style={{ opacity: 0.6 }}>
                       {results.length > 0 ? `Phát hiện: ${results.length} người` : "Đang chờ..."}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Kết quả FaceBox – col-lg-4 */}
            <div className="col-lg-4">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <h5 className="mb-0 text-light">Kết quả hiện tại</h5>
              </div>

              {results.length === 0 && (
                <div className="alert alert-secondary py-2 small mb-3">
                  Chưa nhận diện được khuôn mặt nào. Hãy nhìn thẳng vào camera.
                </div>
              )}

              <div className="row g-3" style={{maxHeight: '70vh', overflowY: 'auto'}}>
                {results.map((face, index) => (
                  <div key={index} className="col-12">
                    <FaceBox face={face} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default WebcamPage;