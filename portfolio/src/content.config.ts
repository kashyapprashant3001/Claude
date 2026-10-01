import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const work = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/work' }),
  schema: z.object({
    title: z.string(),
    client: z.string(),
    year: z.number(),
    role: z.string(),
    summary: z.string(),
    tags: z.array(z.string()).default([]),
    cover: z.string(),
    order: z.number().default(0),
  }),
});

const films = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/films' }),
  schema: z.object({
    title: z.string(),
    logline: z.string(),
    year: z.number(),
    runtime: z.string(),
    role: z.string(),
    poster: z.string(),
    // Either a self-hosted file (src) or a Vimeo/YouTube embed URL (embed).
    src: z.string().optional(),
    embed: z.string().url().optional(),
    credits: z.array(z.object({ role: z.string(), name: z.string() })).default([]),
    featured: z.boolean().default(false),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

export const collections = { work, films, blog };
