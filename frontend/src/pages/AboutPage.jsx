import React from "react";

const AboutPage = () => {
  return (
    <>
      {/* Nhúng CSS trực tiếp vào component để đảm bảo giao diện giống index.html 
        mà không cần tạo file CSS riêng. 
      */}
      <style>
        {`
          :root {
            --primary-color: #0dcaf0;
          }
          .about-page-wrapper {
            background-color: #15171b;
            color: #f8f9fa;
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          }
          .hero-section {
            padding: 80px 0;
            background: linear-gradient(180deg, rgba(21,23,27,0.9) 0%, rgba(13,202,240,0.05) 100%);
            border-bottom: 1px solid #2d3035;
          }
          .feature-card {
            background-color: #212529;
            border: 1px solid #343a40;
            transition: transform 0.3s ease, border-color 0.3s ease;
            height: 100%;
          }
          .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary-color);
          }
          .btn-glow {
            box-shadow: 0 0 15px rgba(13, 202, 240, 0.4);
            transition: box-shadow 0.3s ease;
          }
          .btn-glow:hover {
            box-shadow: 0 0 25px rgba(13, 202, 240, 0.6);
          }
          .badge-custom {
            font-weight: 500;
            padding: 8px 12px;
          }
          /* Custom table styles to match HTML */
          .table-dark {
            --bs-table-bg: #212529;
            --bs-table-border-color: #373b3e;
          }
        `}
      </style>

      <div className="about-page-wrapper">

        {/* Main Content Section */}
        <section className="py-5">
          <div className="container">
            {/* Overview Card */}
            <div className="row g-4 mb-5">
              <div className="col-12">
                <div className="card bg-dark border-secondary">
                  <div className="card-body p-4">
                    <h3 className="card-title text-light mb-3">
                      <i className="ti-info-alt text-info me-2"></i>Tổng quan
                    </h3>
                    <p className="card-text text-light opacity-75">
                      Dự án này xây dựng một hệ thống nhận diện khuôn mặt
                      end-to-end. Thay vì sử dụng phương pháp so khớp tuyến tính
                      (Brute-force) truyền thống tốn kém tài nguyên, chúng tôi áp
                      dụng cấu trúc dữ liệu đồ thị <strong>HNSW</strong> để tìm
                      kiếm vector láng giềng gần nhất (ANN), giúp hệ thống có khả
                      năng mở rộng với dữ liệu lớn mà vẫn giữ độ trễ thấp.
                    </p>
                    <div className="d-flex flex-wrap gap-2 mt-3">
                      <span className="badge bg-primary badge-custom">
                        <i className="ti-loop me-1"></i> HNSW vector search
                      </span>
                      <span className="badge bg-success badge-custom">
                        <i className="ti-face-smile me-1"></i> Real-time ID
                      </span>
                      <span className="badge bg-light text-dark badge-custom">
                        <i className="ti-layers-alt me-1"></i> React &amp; Flask
                      </span>
                      <span className="badge bg-secondary badge-custom">
                        <i className="ti-harddrives me-1"></i> MongoDB
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Features Card */}
              <div className="col-md-6">
                <div className="card feature-card">
                  <div className="card-body p-4">
                    <h4 className="card-title text-info mb-3">
                      <i className="ti-bolt me-2"></i>Tính năng chính
                    </h4>
                    <ul className="text-light opacity-75">
                      <li className="mb-2">
                        Nhận diện khuôn mặt realtime qua Webcam.
                      </li>
                      <li className="mb-2">
                        Hỗ trợ upload ảnh để tìm kiếm trong CSDL.
                      </li>
                      <li className="mb-2">
                        So sánh hiệu năng giữa HNSW và Brute-force.
                      </li>
                      <li className="mb-2">
                        Quản lý Metadata sinh viên (MSSV, Tên) qua MongoDB.
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Workflow Card */}
              <div className="col-md-6">
                <div className="card feature-card">
                  <div className="card-body p-4">
                    <h4 className="card-title text-info mb-3">
                      <i className="ti-direction-alt me-2"></i>Luồng xử lý
                    </h4>
                    <ol className="text-light opacity-75">
                      <li className="mb-2">
                        Client gửi ảnh (Base64/File) lên API.
                      </li>
                      <li className="mb-2">
                        Server dùng <code>face_recognition</code> để encode ra
                        vector 128 chiều.
                      </li>
                      <li className="mb-2">
                        Thuật toán HNSW duyệt đồ thị để tìm vector tương đồng
                        nhất.
                      </li>
                      <li className="mb-2">
                        Trả về thông tin sinh viên và độ tin cậy.
                      </li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            {/* Team Section */}
            <div className="mb-5">
              <h3 className="text-center mb-4 text-uppercase fw-bold text-light">
                Nhóm thực hiện
              </h3>
              <div className="card bg-dark border-secondary">
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-dark table-hover mb-0 align-middle text-center">
                      <thead className="table-active">
                        <tr>
                          <th className="py-3">Họ và Tên</th>
                          <th className="py-3">MSSV</th>
                          <th className="py-3">Vai trò</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="fw-bold text-info">Lê Hoàng Long</td>
                          <td>2411915</td>
                          <td>
                            <span className="badge bg-danger me-2">Backend</span>
                            HNSW &amp; API
                          </td>
                        </tr>
                        <tr>
                          <td className="fw-bold text-info">Nguyễn Tiến Đạt</td>
                          <td>2410712</td>
                          <td>
                            <span className="badge bg-primary me-2">Frontend</span>
                            Web Interface
                          </td>
                        </tr>
                        <tr>
                          <td className="fw-bold text-info">Nguyễn Hoàng Minh</td>
                          <td>2412084</td>
                          <td>
                            <span className="badge bg-success me-2">Data</span>
                            Pipeline &amp; MongoDB
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              <p className="text-center text-secondary mt-3 small">
                Dự án thuộc môn <strong>Data Structures &amp; Algorithms</strong>{" "}
                – Chương trình Tài năng (Honors Program) - HCMUT.
              </p>
            </div>
          </div>
        </section>
      </div>
    </>
  );
};

export default AboutPage;