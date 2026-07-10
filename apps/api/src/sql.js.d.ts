declare module "sql.js" {
  interface Database {
    run(sql: string, params?: any[]): Database;
    exec(sql: string, params?: any[]): QueryExecResult[];
    prepare(sql: string): Statement;
    export(): Uint8Array;
    close(): void;
  }
  interface QueryExecResult {
    columns: string[];
    values: Array<Array<number | string | null>>;
  }
  interface Statement {
    bind(params?: any[]): boolean;
    step(): boolean;
    getAsObject(params?: object): Record<string, any>;
    free(): boolean;
  }
  interface SqlJsStatic {
    Database: new (data?: ArrayLike<number> | Buffer | null) => Database;
  }
  export { Database, SqlJsStatic, Statement, QueryExecResult };
  export default function initSqlJs(config?: any): Promise<SqlJsStatic>;
}