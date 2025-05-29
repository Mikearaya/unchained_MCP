// exportSchema.js
import { getIntrospectionQuery, buildClientSchema, printSchema } from 'graphql';
import { GraphQLClient } from 'graphql-request';
import fs from 'fs';

const endpoint = 'http://localhost:4010/graphql';

const introspectAndSave = async () => {
  const introspectionQuery = getIntrospectionQuery({ descriptions: true });
  const data = await new GraphQLClient(endpoint).request(introspectionQuery);
  const schema = buildClientSchema(data);
  const sdl = printSchema(schema);
  fs.writeFileSync('schema.graphql', sdl);
};

introspectAndSave();
