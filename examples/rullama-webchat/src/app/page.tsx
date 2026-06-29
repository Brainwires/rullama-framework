import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { verifyJwt } from "@/lib/jwt";
import ChatPane from "@/components/ChatPane";

export const dynamic = "force-dynamic";

export default async function Home() {
  const secret = process.env.WEBCHAT_SECRET;
  const wsBase = process.env.NEXT_PUBLIC_GATEWAY_WS ?? "ws://localhost:18789";

  const cookieStore = await cookies();
  const token = cookieStore.get("webchat_jwt")?.value;

  if (!secret || !token || !verifyJwt(token, secret)) {
    redirect("/login");
  }

  return <ChatPane wsBase={wsBase} />;
}
