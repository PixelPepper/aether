import { defineConfig } from 'vite';

export default defineConfig( {
	base: './',
	build: {
		target: 'esnext',
		chunkSizeWarningLimit: 4000,
		assetsInlineLimit: 0,
	},
	esbuild: { target: 'esnext' },
	optimizeDeps: { esbuildOptions: { target: 'esnext' } },
} );
