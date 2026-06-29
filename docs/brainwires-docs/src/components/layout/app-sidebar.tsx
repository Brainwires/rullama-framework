"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { ChevronRight } from "lucide-react";
import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupContent,
  SidebarGroupLabel, SidebarHeader, SidebarMenu, SidebarMenuButton,
  SidebarMenuItem, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { NAV_TREE, type NavItem } from "@/lib/nav";

function NavLeaf({ item }: { item: NavItem }) {
  const pathname = usePathname();
  const isActive = item.href === "/" ? pathname === "/" : pathname === item.href;
  return (
    <SidebarMenuSubItem>
      <SidebarMenuSubButton render={<Link href={item.href!} />} isActive={isActive}>
        {item.title}
      </SidebarMenuSubButton>
    </SidebarMenuSubItem>
  );
}

function NavGroup({ item }: { item: NavItem }) {
  const pathname = usePathname();
  const isChildActive = item.children?.some(
    (c) => c.href && pathname.startsWith(c.href) && c.href !== "/"
  );
  const [open, setOpen] = useState(isChildActive ?? false);

  if (!item.children) {
    const isActive = item.href === "/" ? pathname === "/" : pathname === item.href;
    return (
      <SidebarMenuItem>
        <SidebarMenuButton render={<Link href={item.href!} />} isActive={isActive}>
          {item.title}
        </SidebarMenuButton>
      </SidebarMenuItem>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <SidebarMenuItem>
        <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors">
          <span className="flex-1 text-left">{item.title}</span>
          <ChevronRight className={`size-4 shrink-0 transition-transform ${open ? "rotate-90" : ""}`} />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <SidebarMenuSub>
            {item.children.map((child) => (
              <NavLeaf key={child.href ?? child.title} item={child} />
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuItem>
    </Collapsible>
  );
}

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader className="border-b px-4 py-3">
        <Link href="/" className="flex items-center gap-0.5 font-semibold hover:opacity-80 transition-opacity">
          <span className="text-lg tracking-tight">brainwires</span>
          <span className="text-muted-foreground text-sm font-normal">.dev</span>
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Documentation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_TREE.map((item) => (
                <NavGroup key={item.href ?? item.title} item={item} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
