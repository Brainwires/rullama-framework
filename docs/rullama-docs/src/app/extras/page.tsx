import type { Metadata } from "next";
import Link from "next/link";
import { EXTRAS_LIST } from "@/lib/crates";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const metadata: Metadata = { title: "Extras" };

export default function ExtrasPage() {
  return (
    <div className="px-6 py-8 max-w-[1400px] mx-auto w-full">
      <h1 className="text-3xl font-bold mb-2">Extras</h1>
      <p className="text-muted-foreground mb-8">Standalone applications, tools, and demos built on the Brainwires Framework.</p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {EXTRAS_LIST.map((extra) =>
          extra.hasReadme ? (
            <Link key={extra.name} href={`/extras/${extra.name}`} className="group block">
              <Card className="h-full transition-shadow hover:shadow-md">
                <CardHeader className="pb-2">
                  <CardTitle className="font-mono text-sm font-semibold group-hover:text-primary transition-colors">{extra.name}</CardTitle>
                </CardHeader>
                <CardContent><p className="text-muted-foreground text-sm leading-relaxed">{extra.description}</p></CardContent>
              </Card>
            </Link>
          ) : (
            <Card key={extra.name} className="h-full opacity-70">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <CardTitle className="font-mono text-sm font-semibold">{extra.name}</CardTitle>
                  <Badge variant="outline" className="text-[10px]">no readme</Badge>
                </div>
              </CardHeader>
              <CardContent><p className="text-muted-foreground text-sm leading-relaxed">{extra.description}</p></CardContent>
            </Card>
          )
        )}
      </div>
    </div>
  );
}
