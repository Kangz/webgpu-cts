export const description = `
Tests for queries/filtering, loading, and running.
`;

import {
  DefaultFixture,
  objectEquals,
  paramsEquals,
  RunCase,
  TestGroup,
} from '../../framework/index.js';
import { IGroupDesc, IListing, ITestNode, TestLoader } from '../../framework/loader.js';
import { Logger } from '../../framework/logger.js';

const listingData: { [k: string]: IGroupDesc[] } = {
  suite1: [
    { path: '', description: 'desc 1a' },
    { path: 'foo', description: 'desc 1b' },
    { path: 'bar/', description: 'desc 1c' },
    { path: 'bar/buzz', description: 'desc 1d' },
    { path: 'baz', description: 'desc 1e' },
  ],
  suite2: [{ path: '', description: 'desc 2a' }, { path: 'foof', description: 'desc 2b' }],
};

const nodesData: { [k: string]: ITestNode } = {
  'suite1/README.txt': { description: 'desc 1a' },
  'suite1/foo.spec.js': {
    description: 'desc 1b',
    g: (() => {
      const g = new TestGroup(DefaultFixture);
      g.test('hello', () => {});
      g.test('bonjour', () => {});
      g.test('hola', () => {});
      return g;
    })(),
  },
  'suite1/bar/README.txt': { description: 'desc 1c' },
  'suite1/bar/buzz.spec.js': {
    description: 'desc 1d',
    g: (() => {
      const g = new TestGroup(DefaultFixture);
      g.test('zap', () => {});
      return g;
    })(),
  },
  'suite1/baz.spec.js': {
    description: 'desc 1e',
    g: (() => {
      const g = new TestGroup(DefaultFixture);
      g.test('zed', () => {}).params([
        { a: 1, b: 2 }, //
        { a: 1, b: 3 },
      ]);
      return g;
    })(),
  },
  'suite2/foof.spec.js': {
    description: 'desc 2b',
    g: (() => {
      const g = new TestGroup(DefaultFixture);
      g.test('blah', t => {
        t.ok();
      });
      g.test('bleh', t => {
        t.ok();
        t.ok();
      }).params([{}]);
      return g;
    })(),
  },
};

export class TestTestLoader extends TestLoader {
  protected async listing(suite: string): Promise<IListing> {
    return { suite, groups: listingData[suite] };
  }

  protected async import(path: string): Promise<ITestNode> {
    if (!nodesData.hasOwnProperty(path)) {
      throw new Error('[test] mock file ' + path + ' does not exist');
    }
    return nodesData[path];
  }
}

class LoadingTest extends DefaultFixture {
  loader: TestTestLoader = new TestTestLoader();

  async load(filters: string[]) {
    const listing = await this.loader.loadTests('./out', filters);
    const entries = Promise.all(
      Array.from(listing, ({ suite, path, node }) =>
        node.then((n: ITestNode) => ({ suite, path, node: n }))
      )
    );
    return entries;
  }

  async singleGroup(query: string): Promise<RunCase[]> {
    const [, rec] = new Logger().record('');
    const a = await this.load([query]);
    if (a.length !== 1) {
      throw new Error('more than one group');
    }
    const g = a[0].node.g;
    if (g === undefined) {
      throw new Error('group undefined');
    }
    return Array.from(g.iterate(rec));
  }
}

export const g = new TestGroup(LoadingTest);

g.test('whole suite', async t => {
  t.expect((await t.load(['suite1'])).length === 5);
  t.expect((await t.load(['suite1:'])).length === 5);
});

g.test('partial suite', async t => {
  t.expect((await t.load(['suite1:f'])).length === 1);
  t.expect((await t.load(['suite1:fo'])).length === 1);
  t.expect((await t.load(['suite1:foo'])).length === 1);
  t.expect((await t.load(['suite1:foof'])).length === 0);
  t.expect((await t.load(['suite1:ba'])).length === 3);
  t.expect((await t.load(['suite1:bar'])).length === 2);
  t.expect((await t.load(['suite1:bar/'])).length === 2);
  t.expect((await t.load(['suite1:bar/b'])).length === 1);
});

g.test('whole group', async t => {
  await t.shouldReject(t.load(['suite1::']));
  await t.shouldReject(t.load(['suite1:bar:']));
  await t.shouldReject(t.load(['suite1:bar/:']));
  t.expect((await t.singleGroup('suite1:bar/buzz:')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:')).length === 2);

  {
    const foo = (await t.load(['suite1:foo:']))[0];
    t.expect(foo.suite === 'suite1');
    t.expect(foo.path === 'foo');
    if (foo.node.g === undefined) {
      throw new Error('foo group');
    }
    const [, rec] = new Logger().record('');
    t.expect(Array.from(foo.node.g.iterate(rec)).length === 3);
  }
});

g.test('partial group', async t => {
  t.expect((await t.singleGroup('suite1:foo:h')).length === 2);
  t.expect((await t.singleGroup('suite1:foo:he')).length === 1);
  t.expect((await t.singleGroup('suite1:foo:hello')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:zed')).length === 2);
});

g.test('whole test', async t => {
  t.expect((await t.singleGroup('suite1:foo:hello:')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:zed:')).length === 0);
  t.expect((await t.singleGroup('suite1:baz:zed:')).length === 0);
  t.expect((await t.singleGroup('suite1:baz:zed:{}')).length === 0);
  t.expect((await t.singleGroup('suite1:baz:zed:{"a":1,"b":2}')).length === 1);
});

g.test('partial test', async t => {
  t.expect((await t.singleGroup('suite1:baz:zed~')).length === 2);
  t.expect((await t.singleGroup('suite1:baz:zed~{}')).length === 2);
  t.expect((await t.singleGroup('suite1:baz:zed~{"a":1}')).length === 2);
  t.expect((await t.singleGroup('suite1:baz:zed~{"a":1,"b":2}')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:zed~{"b":2,"a":1}')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:zed~{"b":2}')).length === 1);
  t.expect((await t.singleGroup('suite1:baz:zed~{"a":2}')).length === 0);
  t.expect((await t.singleGroup('suite1:baz:zed~{"c":3}')).length === 0);
});

g.test('end2end', async t => {
  const l = await t.load(['suite2:foof']);
  if (l.length !== 1) {
    throw new Error('listing length');
  }

  t.expect(l[0].suite === 'suite2');
  t.expect(l[0].path === 'foof');
  t.expect(l[0].node.description === 'desc 2b');
  if (l[0].node.g === undefined) {
    throw new Error();
  }
  t.expect(l[0].node.g.iterate instanceof Function);

  const log = new Logger();
  const [res, rec] = log.record(l[0].path);
  const rcs = Array.from(l[0].node.g.iterate(rec));
  if (rcs.length !== 2) {
    throw new Error('iterate length');
  }

  t.expect(rcs[0].testcase.name === 'blah');
  t.expect(rcs[0].testcase.params === null);

  t.expect(rcs[1].testcase.name === 'bleh');
  t.expect(paramsEquals(rcs[1].testcase.params, {}));

  t.expect(log.results[0] === res);
  t.expect(res.path === 'foof');
  t.expect(res.cases.length === 0);

  {
    const cases = res.cases;
    const res0 = await rcs[0].run();
    if (cases.length !== 1) {
      throw new Error('results cases length');
    }
    t.expect(res.cases[0] === res0);
    t.expect(res0.name === 'blah');
    t.expect(res0.params === null);
    t.expect(res0.status === 'pass');
    t.expect(res0.timems > 0);
    if (res0.logs === undefined) {
      throw new Error('results case logs');
    }
    t.expect(objectEquals(res0.logs, ['OK']));
  }
  {
    // Store cases off to a separate variable due to a typescript bug
    // where it can't detect that res.cases.length might change between
    // above and here.
    const cases = res.cases;
    const res1 = await rcs[1].run();
    if (cases.length !== 2) {
      throw new Error('results cases length');
    }
    t.expect(res.cases[1] === res1);
    t.expect(res1.name === 'bleh');
    t.expect(paramsEquals(res1.params, {}));
    t.expect(res1.status === 'pass');
    t.expect(res1.timems > 0);
    if (res1.logs === undefined) {
      throw new Error('results case logs');
    }
    t.expect(objectEquals(res1.logs, ['OK', 'OK']));
  }
});
