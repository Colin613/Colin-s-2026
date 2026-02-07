import { redirect } from "next/navigation";

export default function HomePage() {
  // 重定向到集成应用页面
  redirect("/integrated");
}
