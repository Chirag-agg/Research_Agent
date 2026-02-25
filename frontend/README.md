# Deep Research Agent - Frontend

Next.js 14 frontend for the Deep Research Agent with production architecture generation capabilities.

## Features

- 🔍 Interactive research interface
- 🏗️ Architecture plan visualization
- 📊 Evidence graph display
- 🎨 Modern UI with WebGL backgrounds
- 📱 Responsive design

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm (recommended) or npm

### Installation

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

- `app/` - Next.js 14 App Router pages
  - `page.tsx` - Homepage
  - `research/page.tsx` - Research interface
- `components/` - React components
  - `architecture-plan.tsx` - Architecture display
  - `research-plan.tsx` - Research plan UI
  - `ui/` - shadcn/ui components
- `lib/` - Utility functions

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI)
- **Animations**: OGL WebGL renderer
- **Markdown**: react-markdown with remark-gfm

## Development

```bash
# Development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start

# Lint
pnpm lint
```

## Deployment

Deploy on Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

Or deploy manually:
```bash
pnpm build
pnpm start
```

---

Built with ❤️ using Next.js and React

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
