import readline from 'readline';
import { Ollama } from '@langchain/community/llms/ollama';
import { BufferMemory } from 'langchain/memory';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { DynamicTool } from '@langchain/core/tools';
import { request } from 'graphql-request';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise(resolve => rl.question(q, resolve));

const vectorStore = await HNSWLib.load('./vectorstore', new OllamaEmbeddings({ model: 'nomic-embed-text' }));

const graphqlEndpoint = 'http://localhost:4010/graphql'; 

const runGraphQLTool = new DynamicTool({
  name: 'runGraphQLQuery',
  description: 'Runs a GraphQL query against the backend and returns the result.',
  func: async (query) => {

    try {
        const cleanedQuery = query
        .replace(/```graphql\n?/g, '')
        .replace(/```/g, '')
        .trim();

      const result = await request(graphqlEndpoint, cleanedQuery);
      return JSON.stringify(result, null, 2);
    } catch (e) {
      return `GraphQL Error: ${e.message}`;
    }
  },
});

const llm = new Ollama({ model: 'phi4-mini' });
const memory = new BufferMemory();

const agentExecutor = await initializeAgentExecutorWithOptions(
  [runGraphQLTool],
  llm,
  {
    agentType: 'zero-shot-react-description',
    verbose: true,
  }
);

async function run() {
  while (true) {
    const userInput = await ask('\n\nPrompt> ');
    if (userInput.toLowerCase() === 'exit') break;

    const relatedDocs = await vectorStore.similaritySearch(userInput, 10);
    const context = relatedDocs.map(d => d.pageContent).join('\n');

    if (!context) {
      console.log("⚠️ Couldn't find relevant schema context.");
      continue;
    }

    const history = await memory.loadMemoryVariables({});

    const agentInput = `
You are a GraphQL assistant.
Use the schema context and conversation history to generate a valid GraphQL query or mutation.

Conversation history:
${history.chat_history ?? ''}

Schema context:
${context}

User request:
${userInput}

Generate a GraphQL query and run it using the "runGraphQLQuery" tool.
`;

    const result = await agentExecutor.invoke({ input: agentInput });
    console.log('\n Assistant>\n', result.output);

    await memory.saveContext({ input: userInput }, { output: result.output });
  }

  rl.close();
}

run();
