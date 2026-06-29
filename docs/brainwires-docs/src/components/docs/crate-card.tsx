"use client";

import Link from "next/link";
import { ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { type CrateMeta, TIER_LABELS, TIER_COLORS } from "@/lib/crates";
import { cn } from "@/lib/utils";

export function CrateCard({ crate, href }: { crate: CrateMeta; href: string }) {
  return (
    <Link href={href} className="group block">
      <Card className="h-full transition-shadow hover:shadow-md">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="font-mono text-sm font-semibold group-hover:text-primary transition-colors leading-snug">
              {crate.name}
            </CardTitle>
            <span className={cn("inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[10px] font-medium", TIER_COLORS[crate.tier])}>
              {TIER_LABELS[crate.tier]}
            </span>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-muted-foreground text-sm leading-relaxed">{crate.description}</p>
          {crate.features && crate.features.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {crate.features.slice(0, 4).map((f) => (
                <Badge key={f} variant="secondary" className="font-mono text-[10px]">{f}</Badge>
              ))}
              {crate.features.length > 4 && <Badge variant="outline" className="text-[10px]">+{crate.features.length - 4}</Badge>}
            </div>
          )}
          <div className="flex items-center gap-3 pt-1">
            <a href={`https://crates.io/crates/${crate.name}`} target="_blank" rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs transition-colors">
              crates.io <ExternalLink className="size-3" />
            </a>
            <a href={`https://docs.rs/${crate.name}`} target="_blank" rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1 text-xs transition-colors">
              docs.rs <ExternalLink className="size-3" />
            </a>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
