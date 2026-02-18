import { ReactNode } from "react";
import DocsLayout from "@/components/docs/DocsLayout";

interface DocsLayoutProps {
    children: ReactNode;
}

export default function Layout({ children }: DocsLayoutProps) {
    return <DocsLayout>{children}</DocsLayout>;
}
