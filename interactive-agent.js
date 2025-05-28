import { Ollama } from '@langchain/community/llms/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import readline from 'readline';

const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });
const vectorStore = await HNSWLib.load('./vectorstore', embeddings); // Load saved store

const llm = new Ollama({ model: 'phi4-mini' });

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function ask(query) {
  return new Promise(resolve => rl.question(query, resolve));
}

async function run() {
  while (true) {
    const prompt = await ask("You> ");
    if (prompt === 'exit') break;

    const results = await vectorStore.similaritySearch(prompt, 10);
    const context = results.map(doc => doc.pageContent).join('\n');

    const finalPrompt = `
You are a GraphQL expert.
Use the following schema to answer questions or generate GraphQL queries.

Schema context:
${context}

User request: ${prompt}

Reply only with a valid GraphQL query or explanation based on the schema provided only.
`;

    const gqlQuery = await llm.invoke(finalPrompt);
    console.log("\nGenerated:\n", gqlQuery);
  }

  rl.close();
}

run();
