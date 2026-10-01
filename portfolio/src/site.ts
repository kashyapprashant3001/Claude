// Edit these to make the site yours.
export const SITE = {
  name: 'Your Name',
  role: 'Designer & filmmaker',
  tagline: 'I design interfaces by day and make short films by night.',
  email: 'hello@example.com',
  location: 'Somewhere, Earth',
  socials: [
    { label: 'Instagram', href: 'https://instagram.com/' },
    { label: 'Vimeo', href: 'https://vimeo.com/' },
    { label: 'LinkedIn', href: 'https://linkedin.com/' },
  ],
  // Background reel. Always muted (browsers only autoplay muted video).
  background: {
    mp4: '/media/bg-loop.mp4', // 720p, desktop
    mobile: '/media/bg-loop-mobile.mp4', // 480p, small screens
    poster: '/media/bg-poster.jpg', // first frame: shown while loading / when motion is reduced
  },
};

export const NAV = [
  { href: '/work', label: 'Work' },
  { href: '/films', label: 'Films' },
  { href: '/blog', label: 'Journal' },
  { href: '/about', label: 'About' },
  { href: '/contact', label: 'Contact' },
];
