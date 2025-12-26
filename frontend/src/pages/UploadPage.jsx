// src/pages/UploadPage.jsx
import React, { useState, useRef } from "react";
import { recognizeImage } from "../services/api";
import FaceBox from "../components/FaceBox";

const UploadPage = () => {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const fileInputRef = useRef(null);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  async function handleUpload() {
    if (!imageFile) return;
    setLoading(true);
    setResults([]);

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const res = await recognizeImage(formData);
      setResults(res?.faces || []);
      setElapsedTime(res?.elapsed_ms || 0);
    } catch (err) {
      console.error(err);
      alert("Có lỗi xảy ra khi gọi API nhận diện.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="d-flex flex-column min-vh-100 bg-dark text-light">
      <main className="flex-grow-1 py-3">
        <div className="container">
          <div className="mb-4">
            <h2 className="h4 mb-1 text-light">Chế độ Upload</h2>
            <p className="small text-light mb-0" style={{ opacity: 0.8 }}>
              Tải một ảnh lên để hệ thống nhận diện các khuôn mặt và ánh xạ sang
              sinh viên tương ứng trong cơ sở dữ liệu.
            </p>
          </div>

          <div className="row g-4">
            {/* Upload and preview section */}
            <div className="col-lg-8">
              <div className="card bg-dark border-secondary h-100">
                <div className="card-body d-flex flex-column">
                  <h5 className="card-title mb-3 text-light">Select image</h5>

                  {/* Larger file selection area */}
                  <div className="mb-3">
                    <label
                      className="form-label text-light mb-2"
                      style={{ opacity: 0.8 }}
                    >
                      Tệp ảnh đầu vào
                    </label>
                    <div
                      className="upload-dropzone d-flex flex-column justify-content-center align-items-center p-4 border border-secondary border-dashed rounded text-light text-center"
                      style={{
                        cursor: "pointer",
                        backgroundColor: "#15171b",
                        minHeight: 220,
                      }}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <i className="ti-image mb-3" style={{ fontSize: 32 }} />
                      <span
                        className="small text-light mb-1"
                        style={{ opacity: 0.8 }}
                      >
                        {imageFile
                          ? `Đã chọn: ${imageFile.name}`
                          : "Nhấn để chọn ảnh từ máy của bạn"}
                      </span>
                      <span
                        className="small text-light"
                        style={{ opacity: 0.5 }}
                      >
                        Hỗ trợ JPG, PNG, JPEG…
                      </span>
                    </div>

                    {/* Hidden actual input */}
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="d-none"
                      onChange={handleFileChange}
                    />
                  </div>

                  {/* Image preview */}
                  {previewUrl && (
                    <div className="mb-3 text-center">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="img-fluid rounded border border-secondary"
                        style={{ maxHeight: 260, objectFit: "contain" }}
                      />
                    </div>
                  )}

                  <button
                    type="button"
                    className="btn btn-primary w-100 rounded-pill fw-semibold mt-auto"
                    onClick={handleUpload}
                    disabled={!imageFile || loading}
                  >
                    {loading ? "Đang xử lý..." : "Nhận diện khuôn mặt"}
                  </button>

                  <p
                    className="small text-light mt-2 mb-0"
                    style={{ opacity: 0.5 }}
                  >
                    Ảnh có thể chứa nhiều khuôn mặt; kết quả chi tiết sẽ hiển thị
                    ở khung bên phải.
                  </p>
                </div>
              </div>
            </div>

            {/* Results – reduced to col-lg-4 */}
            <div className="col-lg-4">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <h5 className="mb-0 text-light" style={{ minWidth: "9em" }}>Kết quả nhận diện</h5>
                <span className="small text-light" style={{ opacity: 0.7 }}>
                  {results.length > 0
                    ? `${results.length} khuôn mặt được nhận diện sau ${Number(elapsedTime).toFixed(2)}ms`
                    : "Chưa có kết quả"}
                </span>
              </div>

              {loading && (
                <div className="alert alert-info py-2 small mb-3">
                  Đang gửi ảnh lên server và xử lý…
                </div>
              )}

              {!loading && results.length === 0 && (
                <div className="alert alert-secondary py-2 small mb-3">
                  Chưa có khuôn mặt nào được nhận diện. Hãy chọn ảnh và bấm{" "}
                  <strong>Nhận diện khuôn mặt</strong>.
                </div>
              )}

              <div className="row g-3">
                {results.map((face, index) => (
                  <div key={index} className="col-sm-6 col-md-12">
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

export default UploadPage;
