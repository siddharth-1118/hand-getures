This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy

This project is split into:

- A Next.js frontend that should be deployed to Vercel
- A Python FastAPI backend that should be deployed to Render

### 1. Deploy the backend to Render

Create a new **Web Service** in Render using this repository and set the root directory to `backend`.

Use these settings:

- Environment: `Python 3`
- Build command: `pip install -r requirements-render.txt`
- Start command: `uvicorn index:app --host 0.0.0.0 --port $PORT`

After deployment, confirm the backend is working by opening the Render URL in your browser. You should see a JSON response from `/`.

This repository includes a `runtime.txt` file to pin Render to Python 3.11, which is compatible with the deployed backend dependencies.

### 2. Deploy the frontend to Vercel

Import the same repository into Vercel and leave the root directory empty.

Add this environment variable in Vercel before deploying:

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-render-service.onrender.com
```

Do not include a trailing slash.

Then deploy the project.

### 3. Local development

For local development, create `.env.local` with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```
