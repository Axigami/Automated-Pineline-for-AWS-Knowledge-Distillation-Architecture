# Cấu hình môi trường & Supabase (cho developer)

Tài liệu này giúp đồng đội clone repo, cấu hình biến môi trường và tạo tài khoản đăng nhập trên Supabase để chạy frontend **Synapse IoT** trên máy local.

---

## 1. Yêu cầu trước khi bắt đầu

| Công cụ | Ghi chú |
|--------|---------|
| **Node.js** | Khuyến nghị LTS (ví dụ 20.x trở lên) |
| **npm** | Đi kèm Node |
| Tài khoản **Supabase** | Miễn phí tại [supabase.com](https://supabase.com) |

---

## 2. Cài đặt dự án trên máy

```bash
git clone <URL-repo-của-team>
cd synapse-iot
npm install
```

---

## 3. File cấu hình biến môi trường (bắt buộc cho auth & API)

### 3.1 Tạo file `.env`

1. Copy file mẫu:

   ```bash
   copy .env.example .env
   ```

   (Trên macOS/Linux: `cp .env.example .env`)

2. Mở `.env` và điền giá trị thật (không commit file `.env` lên Git).

### 3.2 Các biến frontend (Vite)

Ứ dụng dùng **Vite**: chỉ các biến có tiền tố `VITE_` mới được nhúng vào bundle chạy trên trình duyệt.

| Biến | Bắt buộc? | Mô tả |
|------|-----------|--------|
| `VITE_SUPABASE_URL` | **Có** | URL project Supabase (dạng `https://xxxx.supabase.co`) |
| `VITE_SUPABASE_ANON_KEY` | **Có** | **Anon / public key** — dùng ở client; quyền thật do **RLS** trên Supabase quyết định |
| `GEMINI_API_KEY` | Tùy feature | Chỉ cần nếu module nào đó gọi Gemini (xem code) |
| `APP_URL` | Tùy | URL deploy / callback nếu có dùng |

**Vị trí code đọc Supabase:** `src/core/lib/supabaseClient.ts` — singleton `createClient` với `persistSession`, `autoRefreshToken`, `flowType: 'pkce'`.

---

## 4. Tạo project Supabase và lấy URL + Anon key

1. Đăng nhập [Supabase Dashboard](https://supabase.com/dashboard).
2. **New project** → chọn org, đặt tên, mật khẩu database, region.
3. Đợi project khởi tạo xong.
4. Vào **Project Settings** (biểu tượng bánh răng) → **API**:
   - **Project URL** → dán vào `VITE_SUPABASE_URL`.
   - **Project API keys** → mục **anon** `public` → dán vào `VITE_SUPABASE_ANON_KEY`.

Không dùng **service_role** key trong file `.env` của frontend (chỉ dùng server/backend, tuyệt đối không đưa vào bundle client).

---

## 5. Tạo tài khoản để đăng nhập app (Email + mật khẩu)

App đăng nhập qua `signInWithPassword` (xem `src/modules/auth/view/pages/LoginPage.tsx`). Bạn cần **ít nhất một user** trong **Supabase Auth** và (tuỳ schema) bản ghi role nếu app đọc từ bảng `users_roles_settings`.

### 5.1 Bật đăng nhập bằng email

1. Trong project Supabase: **Authentication** → **Providers**.
2. Mục **Email** → bật **Enable Email provider**.
3. Phần **Confirm email** (tuỳ môi trường):
   - **Môi trường dev / nội bộ:** có thể tắt “Confirm email” hoặc dùng **Auto Confirm** khi tạo user thủ công (xem bước dưới) để không bị kẹt chờ email.

### 5.2 Tạo user trực tiếp trên Dashboard (nhanh nhất cho team)

1. **Authentication** → **Users**.
2. **Add user** → **Create new user**.
3. Nhập **email** và **mật khẩu** (đủ mạnh theo policy Supabase nếu có).
4. Tuỳ chọn **Auto Confirm User** nếu muốn đăng nhập ngay không cần xác nhận email.
5. Lưu.

Sau đó mở app (`npm run dev`), vào trang login, nhập đúng email/mật khẩu vừa tạo.

### 5.3 Cách khác: đăng ký từ app (nếu bật Sign up)

Nếu team bật **Sign up** (và có UI đăng ký trong tương lai), user có thể tự tạo tài khoản; vẫn phải đảm bảo **Email provider** bật và policy email phù hợp. Hiện tại flow chính trong repo là **đăng nhập** với user đã tạo sẵn trên Dashboard.

### 5.4 Gán vai trò (role) trong database (nếu app cần)

`AuthProvider` đọc role từ bảng `users_roles_settings` (cột `role_code`: `admin`, `operator`, `viewer`). Nếu sau khi đăng nhập không thấy quyền đúng:

- Kiểm tra **RLS** và policy `SELECT` cho user đó.
- Thêm/cập nhật dòng trong `users_roles_settings` khớp `user_id` (UUID lấy từ **Authentication → Users** → chọn user → copy **User UID**).

Chi tiết schema nằm trong Supabase SQL / migration của team — không nằm trong file này.

---

## 6. Chạy ứng dụng

```bash
npm run dev
```

Mở URL Vite in ra (thường `http://localhost:5173`), vào `/login` và thử đăng nhập.

---

## 7. Checklist lỗi thường gặp

| Hiện tượng | Hướng xử lý |
|------------|-------------|
| Lỗi khi build: thiếu `VITE_SUPABASE_*` | Kiểm tra file `.env` ở **thư mục gốc** repo, tên biến đúng, restart `npm run dev`. |
| Đăng nhập báo sai mật khẩu | User chưa tạo trong **Authentication → Users** hoặc nhập sai; kiểm tra đã **Confirm** user chưa. |
| Đăng nhập được nhưng không có dữ liệu / lỗi 403 | Kiểm tra **RLS** trên các bảng (`alerts_all`, `homes`, …) và quyền `anon`. |
| Session mất sau refresh | Thường do chặn cookie/storage trình duyệt; client dùng `localStorage` (xem `supabaseClient.ts`). |

---

## 8. Liên kết file quan trọng trong repo

| Nội dung | Đường dẫn |
|----------|-----------|
| Client Supabase | `src/core/lib/supabaseClient.ts` |
| Auth context | `src/core/auth/AuthProvider.tsx` |
| Trang đăng nhập | `src/modules/auth/view/pages/LoginPage.tsx` |
| Route bảo vệ | `src/core/auth/ProtectedRoute.tsx` |
| Mẫu biến môi trường | `.env.example` |

---

*Tài liệu này mô tả quy trình chuẩn cho team; nếu policy bảo mật công ty khác (VD: bắt buộc SSO), cần bổ sung riêng.*
