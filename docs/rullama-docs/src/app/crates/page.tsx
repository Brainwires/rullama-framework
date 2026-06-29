import type { Metadata } from "next";
import { CRATE_LIST, TIER_LABELS, type CrateTier } from "@/lib/crates";
import { CrateCard } from "@/components/docs/crate-card";

export const metadata: Metadata = { title: "Framework Crates" };

const TIER_ORDER: CrateTier[] = ["facade", "foundation", "intelligence", "hardware", "network", "utility"];

export default function CratesPage() {
  const byTier = TIER_ORDER.map((tier) => ({
    tier, label: TIER_LABELS[tier],
    crates: CRATE_LIST.filter((c) => c.tier === tier),
  })).filter((g) => g.crates.length > 0);

  return (
    <div className="px-6 py-8 max-w-[1400px] mx-auto w-full">
      <h1 className="text-3xl font-bold mb-2">Framework Crates</h1>
      <p className="text-muted-foreground mb-8">20 independently publishable crates that compose into a full AI agent framework.</p>
      {byTier.map(({ tier, label, crates }) => (
        <section key={tier} className="mb-10">
          <h2 className="text-lg font-semibold mb-4 border-b pb-2">{label}</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {crates.map((crate) => <CrateCard key={crate.name} crate={crate} href={`/crates/${crate.name}`} />)}
          </div>
        </section>
      ))}
    </div>
  );
}
