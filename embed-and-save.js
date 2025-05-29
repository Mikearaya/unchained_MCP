import fs from 'fs';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { buildSchema, printType } from 'graphql';

console.log('Step 1: Load schema')

const schemaSource = fs.readFileSync('./schema.graphql', 'utf8');
const schema = buildSchema(schemaSource);
const typeMap = schema.getTypeMap();

const structuredDocs = Object.values(typeMap)
  .filter(t => !t.name.startsWith('__'))
  .map(t => printType(t));
console.log(structuredDocs)

console.log('Step 2: Chunk schema')
const splitter = new RecursiveCharacterTextSplitter({   chunkSize: 1024,
    chunkOverlap: 100 });
const docs = await splitter.createDocuments(structuredDocs);
console.log('Step 3: Embed with Ollama')
const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });

const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
console.log('Step 4: Save vectors to disk')
await vectorStore.save('./vectorstore'); 

