# Hợp đồng giao tiếp & Quy ước kỹ thuật (Hospital Navigation)

## 1. Quy ước lưới bản đồ & Tọa độ (QUAN TRỌNG CHO TV2)
- **Đơn vị:** Mét (`m`), radian (`rad`), giây (`s`).
- **Hệ tọa độ thế giới:** Gốc `(0, 0)` ở góc dưới cùng bên trái, trục X sang phải, trục Y lên trên.
- **Thứ tự chỉ số mảng Grid:** 
  - Shape của mảng numpy là `(n_rows, n_cols) = (200, 400)` tương ứng `(y, x)`.
  - Chỉ số **0-based**. Khi đọc mảng để viết LiDAR hoặc VFH, luôn lưu ý thứ tự là `[row, col]` tương ứng `[y_index, x_index]`.