// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  // Set this to your real domain before deploying (used for canonical/OG URLs).
  site: 'https://example.com',
  prefetch: { prefetchAll: true },
});
