// exportSchema.js
import { getIntrospectionQuery, buildClientSchema, printSchema } from 'graphql';
import { request } from 'graphql-request';
import fs from 'fs';

const endpoint = 'http://localhost:4010/graphql';

const introspectAndSave = async () => {
  const introspectionQuery = getIntrospectionQuery();
  const data = await request(endpoint, introspectionQuery);
  const schema = buildClientSchema(data);
  const sdl = printSchema(schema);
  fs.writeFileSync('schema.graphql', sdl);
};

introspectAndSave();
