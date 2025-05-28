import fs from 'fs';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
console.log('Step 1: Load schema')

const schema = fs.readFileSync('./schema.graphql', 'utf8');
console.log('Step 2: Chunk schema')
const splitter = new RecursiveCharacterTextSplitter({   chunkSize: 800,
    chunkOverlap: 200, });
const docs = await splitter.createDocuments([schema]);
console.log('Step 3: Embed with Ollama')
const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });

const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
console.log('Step 4: Save vectors to disk')
await vectorStore.save('./vectorstore'); // Will create a folder with saved index
