"use client";

import { Moon, Sun, Search } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { SearchDialog } from "@/components/docs/search-dialog";
import { useState } from "react";

export function Topbar() {
  const { theme, setTheme } = useTheme();
  const [searchOpen, setSearchOpen] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-40 flex h-14 items-center gap-2 border-b bg-background px-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="h-4" />
        <div className="flex flex-1 items-center justify-end gap-2">
          <Button
            variant="outline" size="sm"
            className="text-muted-foreground w-48 justify-start gap-2 text-sm"
            onClick={() => setSearchOpen(true)}
          >
            <Search className="size-3.5" />
            Search docs…
            <kbd className="ml-auto pointer-events-none hidden select-none rounded border px-1.5 py-0.5 font-mono text-[10px] sm:inline-flex">
              ⌘K
            </kbd>
          </Button>
          <Button
            variant="ghost" size="icon" aria-label="Toggle theme"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <Sun className="size-4 dark:hidden" />
            <Moon className="hidden size-4 dark:block" />
          </Button>
        </div>
      </header>
      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
    </>
  );
}
