// Turns the Vite build in dist/ into an Artifact page: the host wraps the page in its own
// <!doctype><html><head><body> skeleton, so we strip those wrappers and keep the content.
// Output: dist/aether.html (page) + dist/assets/* (published alongside as supporting files).
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';

const html = readFileSync( 'dist/index.html', 'utf8' );
const head = html.match( /<head>([\s\S]*?)<\/head>/ )[ 1 ];
const body = html.match( /<body>([\s\S]*?)<\/body>/ )[ 1 ];

const keepHead = head
	.split( '\n' )
	.filter( ( l ) => ! /<meta charset|<meta name="viewport"/.test( l ) )
	.join( '\n' );

const titleLine = keepHead.match( /<title>.*<\/title>/ )[ 0 ];
const rest = keepHead.replace( titleLine, '' );

const page = `${titleLine}\n${rest.trim()}\n${body.trim()}\n`;
writeFileSync( 'dist/aether.html', page );

const files = {};
for ( const f of readdirSync( 'dist/assets' ) ) files[ `assets/${f}` ] = `dist/assets/${f}`;
writeFileSync( 'dist/files.json', JSON.stringify( files, null, 2 ) );
console.log( 'wrote dist/aether.html with', Object.keys( files ).length, 'supporting files' );
