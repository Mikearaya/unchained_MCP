import readline from 'readline';
import { Ollama } from '@langchain/community/llms/ollama';
import { PromptTemplate } from '@langchain/core/prompts';
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
    description: 'Runs a GraphQL query string (not JSON) against the backend and returns the result.',
    func: async (rawInput) => {
      try {
        // Clean out markdown syntax like ```graphql
        let cleaned = rawInput
          .replace(/```graphql\n?/gi, '')
          .replace(/```/g, '')
          .trim();
  
    
        if (cleaned.startsWith('{') && cleaned.includes('"query"')) {
          const parsed = JSON.parse(cleaned);
          cleaned = parsed.query;
        }
  
        const result = await request(graphqlEndpoint, cleaned);
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
    verbose: true
  }
);

async function run() {
  while (true) {
    const userInput = await ask('\nYou> ');
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
    console.log('\Assistant>\n', result.output);

    await memory.saveContext({ input: userInput }, { output: result.output });
  }

  rl.close();
}

run();
