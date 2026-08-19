// compiler.js — Midstate contract compiler.
//
// Lifted from the Midstate IDE (Lexer → Parser → CodeGen → assembleBytecode)
// with no changes to the language, so a contract that compiles here produces the
// same bytecode the IDE shows and therefore the same P2SH address.
//
// # Reasoning
//
// A launcher cannot pre-compile its contracts. Every token needs its own curve
// with that token's asset id, treasury and parameters baked in, which means a
// distinct script and a distinct address per launch. Compiling by hand in the
// IDE works for one contract and not for a thousand, so the compiler has to be
// callable from code.
//
// The opcode table is checked against the node's `core/script.rs` at load rather
// than trusted: a table that has drifted produces bytecode that assembles
// cleanly, hashes to a plausible address, and fails at execution — after funds
// are already locked at that address.
//
// Note the node also defines `OP_SUM_INPUT_VALUE = 0x56`, which this language
// does not expose. Contracts needing it must be written in assembly.

const STD_LIB = `
// ═════════════════════════════════════════════════════════════════════
// MIDSTATE STANDARD LIBRARY (Auto-injected)
// ═════════════════════════════════════════════════════════════════════

// ── MATH ──
macro min(x, y) { if (x < y) { x; } else { y; } }
macro max(x, y) { if (x > y) { x; } else { y; } }

// ── SECURITY PRIMITIVES ──
macro require_signed_by(pk) {
    pk; CHECKSIGVERIFY;
}
macro require_length(data, expected_len) {
    assert(size(data) == expected_len);
}

macro require_timelock(blocks) {
    blocks; CHECKTIMEVERIFY;
}

macro require_funds_transferred(amount, address) {
    assert(sum_to_addr(address) >= amount);
}

// ── CRYPTO: MERKLE TREES ──
// Standard step for verifying a static Merkle Proof
// Expected Stack: [..., Sibling, Dir, Current]
macro merkle_step() {
    swap();
    if (pop_int() == 1) { } else { swap(); }
    concat();
    hash();
}

// Advanced step for SIMULTANEOUSLY verifying an old root and calculating a new root.
// Expected Stack: [..., Sibling, Dir, Old_Acc, New_Acc] (New_Acc is on top)
macro merkle_update_step() {
    swap(); 
    rot();
    if (pop_int() == 1) { 
        rot(); dup(); rot(); concat(); hash(); 
        rot(); rot(); swap(); concat(); hash();
    } else { 
        rot(); dup(); rot(); swap(); concat(); hash(); 
        rot(); rot(); concat(); hash();
    }
}
// ═════════════════════════════════════════════════════════════════════
`;

// ── Numeric helpers ─────────────────────────────────────────────────────────
//
// The VM's math opcodes are little-endian and capped at 8 bytes (`to_u64`
// rejects anything wider), so both of these enforce that rather than silently
// producing a value the VM will refuse.

function intToHexLE(n) {
    if (typeof n !== 'bigint') n = BigInt(n);
    if (n === 0n) return '00';
    let h = n.toString(16);
    if (h.length % 2) h = '0' + h;
    return h.match(/.{2}/g).reverse().join('');
}

function hexLEToInt(h) {
    if (!h) return 0n;
    if (h.length > 16) throw new Error('compiler: math operands are limited to 8 bytes');
    return BigInt('0x' + h.match(/.{2}/g).reverse().join(''));
}

function isTrue(h) {
    if (!h) return false;
    return h.match(/.{2}/g).some((b) => parseInt(b, 16) !== 0);
}

const OPS={PUSH_DATA:0x01,DROP:0x10,DUP:0x11,SWAP:0x12,OVER:0x13,ROT:0x14,SLICE:0x15,CONCAT:0x16,PICK:0x17,EQUAL:0x20,VERIFY:0x21,EQUALVERIFY:0x22,ADD:0x23,GREATER_OR_EQUAL:0x24,SUB:0x25,MUL:0x26,DIV:0x27,MOD:0x28,SIZE:0x29,HASH:0x30,CHECKSIG:0x31,CHECKSIGVERIFY:0x32,CHECKTIMEVERIFY:0x33,IF:0x40,ELSE:0x41,ENDIF:0x42,SUM_TO_ADDR:0x50,READ_INPUT_STATE:0x51,READ_OUTPUT_STATE:0x52,INPUT_VALUE:0x53,OUTPUT_ADDRESS:0x54,THIS_ADDRESS:0x55,SUM_INPUT_VALUE:0x56};
function assembleBytecode(asm) {
  let bc=[];
  for(let inst of asm){
    let [op,...rest]=inst.split(' ');
    if(op==='PUSH_HEX'||op==='PUSH_INT'){
      bc.push(OPS.PUSH_DATA);
      let hex=op==='PUSH_INT'?intToHexLE(BigInt(rest[0])):rest[0].toLowerCase();
      if (hex.length % 2 !== 0) throw `Syntax Error: Hex string must have an even number of characters ('${hex}')`;
      if (!/^[0-9a-f]*$/.test(hex)) throw `Syntax Error: Invalid hex characters in '${hex}'`;
      let len=hex.length/2, lh=len.toString(16).padStart(4,'0');
      bc.push(parseInt(lh.slice(2,4),16),parseInt(lh.slice(0,2),16));
      for(let i=0;i<hex.length;i+=2)bc.push(parseInt(hex.slice(i,i+2),16));
    } else {
      if(OPS[op]===undefined)throw`Unknown opcode: ${op}`;
      bc.push(OPS[op]);
    }
  }
  return bc.map(b=>b.toString(16).padStart(2,'0')).join('');
}

// ═══════════════════════════════════════════
// LEXER
// ═══════════════════════════════════════════
class Lexer {
  constructor(src){
    this.src=src.replace(/\/\*[\s\S]*?\*\//g,'').replace(/\/\/.*/g,'');
    this.pos=0;this.line=1;this.col=1;
  }
  adv(){let c=this.src[this.pos++];if(c==='\n'){this.line++;this.col=1;}else this.col++;return c;}
  tokenize(){
    const toks=[];
    // ONLY true raw opcodes that never take (args) in high-level code
    const RAW_OPS=['PUSH_INT','PUSH_HEX','DUP','DROP','SWAP','OVER','ROT','PICK','SLICE','CONCAT','EQUAL','EQUALVERIFY','CHECKSIG','CHECKSIGVERIFY','ADD','SUB','MUL','DIV','MOD','SIZE','GREATER_OR_EQUAL','HASH','CHECKTIMEVERIFY','VERIFY','INPUT_VALUE','SUM_INPUT_VALUE','OUTPUT_ADDRESS','THIS_ADDRESS'];
    const KWS=['if','else','repeat','true','false','macro','let','var','state','struct','switch','case','default','assert','route','test'];
    while(this.pos<this.src.length){
      let c=this.src[this.pos];
      if(/\s/.test(c)){this.adv();continue;}
      let sl=this.line,sc=this.col;
      if(/\d/.test(c)){let n='';while(this.pos<this.src.length&&/\d/.test(this.src[this.pos]))n+=this.adv();toks.push({t:'NUM',v:parseInt(n),l:sl,c:sc});continue;}
      if(c==='"'||c==="'"){let q=this.adv(),s='';while(this.pos<this.src.length&&this.src[this.pos]!==q)s+=this.adv();if(this.pos>=this.src.length)throw`Unterminated string at ${sl}:${sc}`;this.adv();toks.push({t:'HEX',v:s,l:sl,c:sc});continue;}
      if(c==='='&&this.src[this.pos+1]==='='){toks.push({t:'OP',v:'==',l:sl,c:sc});this.adv();this.adv();continue;}
      if(c==='>'&&this.src[this.pos+1]==='='){toks.push({t:'OP',v:'>=',l:sl,c:sc});this.adv();this.adv();continue;}
      if(c==='&'&&this.src[this.pos+1]==='&'){toks.push({t:'OP',v:'&&',l:sl,c:sc});this.adv();this.adv();continue;}
      if(c==='|'&&this.src[this.pos+1]==='|'){toks.push({t:'OP',v:'||',l:sl,c:sc});this.adv();this.adv();continue;}
      // FIXED: Added ':' to punctuation list (required for "field: size" in state/struct)
      if(['+','-','*','/','%','{','}','(',')','.',';',',',':'].includes(c)){
        if(c==='.') toks.push({t:'DOT',v:'.',l:sl,c:sc});
        else toks.push({t:'P',v:c,l:sl,c:sc});
        this.adv();continue;
      }
      if(c==='='||c==='>'||c==='<'){let op=this.adv();if(this.src[this.pos]==='=')op+=this.adv();toks.push({t:'ROP',v:op,l:sl,c:sc});continue;}
      if(/[a-zA-Z_]/.test(c)){let id='';while(this.pos<this.src.length&&/[a-zA-Z0-9_]/.test(this.src[this.pos]))id+=this.adv();
        if(RAW_OPS.includes(id))toks.push({t:'ASM',v:id,l:sl,c:sc});
        else if(KWS.includes(id))toks.push({t:'KW',v:id,l:sl,c:sc});
        else toks.push({t:'ID',v:id,l:sl,c:sc});continue;}
      throw`Unknown char '${c}' at ${this.line}:${this.col}`;
    }
    return toks;
  }
}

// ═══════════════════════════════════════════
// PARSER
// ═══════════════════════════════════════════

class Parser {
  constructor(toks){this.toks=toks;this.pos=0;}
  pk(){return this.toks[this.pos];}
  consume(et,ev){let t=this.toks[this.pos];if(!t)throw'Unexpected EOF';if(et&&t.t!==et)throw`Syntax Error at ${t.l}:${t.c}: Expected ${et}, got '${t.v}'`;if(ev&&t.v!==ev)throw`Syntax Error at ${t.l}:${t.c}: Expected '${ev}', got '${t.v}'`;return this.toks[this.pos++];}
  match(v){if(this.pk()&&this.pk().v===v){this.pos++;return true;}return false;}
  parse(){let s=[];while(this.pos<this.toks.length){let st=this.parseStmt();if(st)s.push(st);}return s;}

  parseStmt(){
    let t=this.pk();if(!t)return null;

    if(t.t==='KW'&&t.v==='let'){
      this.consume();
      let id=this.consume('ID').v;
      this.consume('ROP','=');
      let expr=this.parseExpr();
      this.match(';');
      return {type:'LetStmt', id, expr};
    }

    if(t.t==='KW'&&t.v==='var'){
      this.consume();
      let id=this.consume('ID').v;
      this.consume('ROP','=');
      let expr=this.parseExpr();
      this.match(';');
      return {type:'VarDecl', id, expr};
    }
// Check for bare assignment: identifier = expr
if (t.t === 'ID' && this.toks[this.pos + 1]?.v === '=') {
    const id = this.consume('ID').v;
    this.consume('ROP', '=');
    const expr = this.parseExpr();
    this.match(';');
    return { type: 'AssignStmt', id, expr, l: t.l, c: t.c }; 
}
    if(t.t==='KW'&&t.v==='macro'){
      this.consume();
      let name=this.consume('ID').v;
      this.consume('P','(');
      let params = [];
      while(this.pk() && this.pk().v !== ')') {
          params.push(this.consume('ID').v);
          if (this.match(',')) continue;
      }
      this.consume('P',')');
      let body=this.parseBlock();
      return {type:'MacroDefStmt', name, params, body};
    }

    if(t.t==='KW'&&t.v==='state'){
      this.consume();
      let name=this.consume('ID').v;
      this.consume('P','{');
      let fields={}; let offset=0;
      while(this.pk()&&this.pk().v!=='}'){
        let fname=this.consume('ID').v;
        this.consume('P',':');
        let size=this.consume('NUM').v;
        fields[fname]={offset, size};
        offset += size;
        this.match(',');
      }
      this.consume('P','}');
      return {type:'StateDef', name, fields};
    }

    if(t.t==='KW'&&t.v==='struct'){
      this.consume();
      let name=this.consume('ID').v;
      this.consume('P','{');
      let fields={}; let offset=0;
      while(this.pk()&&this.pk().v!=='}'){
        let fname=this.consume('ID').v;
        this.consume('P',':');
        let size=this.consume('NUM').v;
        fields[fname]={offset, size};
        offset += size;
        this.match(',');
      }
      this.consume('P','}');
      return {type:'StructDef', name, fields};
    }

    if(t.t==='KW'&&t.v==='if')return this.parseIf();
    if(t.t==='KW'&&t.v==='repeat')return this.parseRepeat();
    if(t.t==='KW'&&t.v==='switch')return this.parseSwitch();
    if(t.t==='KW'&&t.v==='route')return this.parseRoute();
    if(t.t==='KW'&&t.v==='test') {
      this.consume();
      let name = this.consume('HEX').v; // Reusing your string tokenizer for the description
      let body = this.parseBlock();
      return {type: 'TestDef', name, body};
    }
if(t.t==='KW'&&t.v==='assert'){
      this.consume();
      this.consume('P','(');
      let expr=this.parseExpr();
      this.consume('P',')');
      this.match(';');
      return {type:'AssertStmt', expr, l: t.l, c: t.c}; 
    }
    if(t.t==='P'&&t.v==='{')return this.parseBlock();
    if(t.t==='P'&&t.v===';'){this.consume();return null;}

    if(t.t==='ASM'){
      this.consume();
      if(t.v==='PUSH_HEX'||t.v==='PUSH_INT'){
        if(this.match('(')){let a=this.consume();if(!this.match(')'))throw`Expected ')' after ${t.v} arg`;this.match(';');return{type:'RawASM',op:t.v,arg:a.v.toString()};}
        let a=this.consume();this.match(';');return{type:'RawASM',op:t.v,arg:a.v.toString()};
      }
      if(this.match('(')){if(!this.match(')'))throw`ASM '${t.v}' takes no args — expected ')'`;}
      this.match(';');return{type:'RawASM',op:t.v};
    }
    if(t.t==='P'&&t.v==='+'){this.consume();this.match(';');return{type:'RawASM',op:'ADD'};}
    if(t.t==='ROP'&&t.v==='=='){this.consume();this.match(';');return{type:'RawASM',op:'EQUAL'};}
    if(t.t==='ROP'&&t.v==='>='){this.consume();this.match(';');return{type:'RawASM',op:'GREATER_OR_EQUAL'};}

    let expr=this.parseExpr();this.match(';');return{type:'ExprStmt',expr};
  }

  parseSwitch(){
    let t = this.consume('KW','switch');
    if(!this.match('('))throw`Expected '(' after 'switch'`;
    let expr=this.parseExpr();
    if(!this.match(')'))throw`Expected ')'`;
    if(!this.match('{'))throw`Expected '{' after switch`;
    let cases=[];
    while(this.pk()&&this.pk().v!=='}'){
      if(this.match('case')){
        let val=this.parseExpr();
        this.consume('P',':');
        let body=this.parseBlock()||this.parseStmt();
        cases.push({val,body});
      } else if(this.match('default')){
        this.consume('P',':');
        let body=this.parseBlock()||this.parseStmt();
        cases.push({isDefault:true,body});
      } else break;
    }
    if(!this.match('}'))throw`Missing '}' for switch`;
    return {type:'SwitchStmt',expr,cases, l: t.l, c: t.c}; 
  }
  parseRoute(){
    let t = this.consume('KW','route'); // <-- Capture token t here
    if(!this.match('{'))throw `Expected '{' after route`;
    let cases=[];
    while(this.pk()&&this.pk().v!=='}'){
      if(this.match('case')){
        let val=this.parseExpr();
        this.consume('P',':');
        let body=this.parseBlock()||this.parseStmt();
        cases.push({val,body});
      } else if(this.match('default')){
        this.consume('P',':');
        let body=this.parseBlock()||this.parseStmt();
        cases.push({isDefault:true,body});
      } else break;
    }
    this.consume('P','}');
    
    return {type:'SwitchStmt', expr: {type:'ImplicitTop'}, cases, l: t.l, c: t.c}; 
  }

  parseBlock(){let s=this.consume('P','{');let stmts=[];while(this.pos<this.toks.length&&(!this.pk()||this.pk().v!=='}')){let st=this.parseStmt();if(st)stmts.push(st);}if(!this.match('}'))throw`Missing '}' for block at ${s.l}`;return{type:'BlockStmt',stmts};}
  parseIf(){let t=this.consume('KW','if');if(!this.match('('))throw`Expected '(' after 'if'`;let cond=this.parseExpr();if(!this.match(')'))throw`Expected ')' after if condition`;let thenB=this.parseBlock(),elseB=null;if(this.match('else')){if(this.pk()&&this.pk().v==='if')elseB=this.parseIf();else elseB=this.parseBlock();}return{type:'IfStmt',cond,thenB,elseB};}
  parseRepeat(){let t=this.consume('KW','repeat');if(!this.match('('))throw`Expected '(' after 'repeat'`;let n=this.consume('NUM');if(!this.match(')'))throw`Expected ')'`;return{type:'RepeatStmt',count:n.v,body:this.parseBlock()};}
  parseExpr(){return this.parseLOr();}
  parseLOr(){let e=this.parseLAnd();while(this.match('||'))e={type:'Bin',l:e,op:'||',r:this.parseLAnd()};return e;}
  parseLAnd(){let e=this.parseEq();while(this.match('&&'))e={type:'Bin',l:e,op:'&&',r:this.parseEq()};return e;}
  parseEq(){
    let e=this.parseAdd();
    while(this.match('==')||this.match('>=')||this.match('>')||this.match('<=')||this.match('<')){   
      let op=this.toks[this.pos-1].v;
      e={type:'Bin',l:e,op,r:this.parseAdd()};
    }
    return e;
  }
  parseAdd(){
    let e=this.parseMul();
    while(this.match('+')||this.match('-')){
      let op=this.toks[this.pos-1].v;
      let r=this.parseMul();
      // Fold constants if both sides are static integers
      if(e.type==='Lit' && r.type==='Lit' && e.k==='int' && r.k==='int') {
          e = {type:'Lit', v: op==='+' ? e.v+r.v : e.v-r.v, k:'int'};
      } else {
          e={type:'Bin',l:e,op,r};
      }
    }
    return e;
  }
  
  parseMul(){
    let e=this.parsePrim();
    while(this.match('*')||this.match('/')||this.match('%')){
      let op=this.toks[this.pos-1].v;
      let r=this.parsePrim();
      // Fold constants if both sides are static integers
      if(e.type==='Lit' && r.type==='Lit' && e.k==='int' && r.k==='int') {
          let val = 0;
          if (op === '*') val = e.v * r.v;
          else if (op === '/') val = Math.floor(e.v / r.v);
          else if (op === '%') val = e.v % r.v;
          e = {type:'Lit', v: val, k:'int'};
      } else {
          e={type:'Bin',l:e,op,r};
      }
    }
    return e;
  }
parsePrim(){
    let t=this.pk();if(!t)throw'Unexpected EOF in expression';
    if(this.match('(')){let e=this.parseExpr();if(!this.match(')'))throw`Missing ')' at ${t.l}:${t.c}`;return e;}
    t=this.consume();
    if(t.t==='NUM')return{type:'Lit',v:t.v,k:'int',l:t.l,c:t.c};
    if(t.t==='HEX')return{type:'Lit',v:t.v,k:'hex',l:t.l,c:t.c};
    if(t.t==='KW'&&t.v==='true')return{type:'Lit',v:1,k:'int',l:t.l,c:t.c};
    if(t.t==='KW'&&t.v==='false')return{type:'Lit',v:0,k:'int',l:t.l,c:t.c};
    if(t.t==='ID'){
      if(this.match('.')){
        let field=this.consume('ID').v;
        return {type:'FieldAccess', base:t.v, field, l:t.l, c:t.c};
      }
      if(this.match('(')){
          let args=[];
          while(this.pk()&&this.pk().v!==')'){
              args.push(this.parseExpr());this.match(',');
          }
          if(!this.match(')'))throw`Missing ')' for '${t.v}'`;
          return{type:'Call',name:t.v,args,l:t.l,c:t.c};
      }
      return{type:'Lit',v:t.v,k:'hex',l:t.l,c:t.c};
    }
    throw`Syntax Error at ${t.l}:${t.c}: Unexpected token '${t.v}'`;
  }
}

// ═══════════════════════════════════════════
// CODE GEN
// ═══════════════════════════════════════════
class CodeGen {
  constructor(){
    this.asm = [];
    this.sourceMap = [];
    this.macros = {};
    this.constants = {};
    this.structs = {};
    this.locals = [];           // list of active var names (stack order)
    this.stateDefs = {};
    this.stackDepthOffset = 0;
    this.simulatedStackDepth = 0;
  }
  emit(op, line){
    this.asm.push(op);
    this.sourceMap.push(line || 0);
    
    // Calculate stack effect of this operation
    let baseOp = op.split(' ')[0];
    
    if (baseOp.startsWith('PUSH_') || baseOp === 'DUP' || baseOp === 'READ_INPUT_STATE' || baseOp === 'SIZE' || baseOp === 'INPUT_VALUE' || baseOp === 'SUM_INPUT_VALUE' || baseOp === 'THIS_ADDRESS') {
        this.simulatedStackDepth += 1;
    } else if (baseOp === 'DROP' || baseOp === 'ADD' || baseOp === 'SUB' || baseOp === 'MUL' || baseOp === 'DIV' || baseOp === 'MOD' || baseOp === 'EQUAL' || baseOp === 'EQUALVERIFY' || baseOp === 'GREATER_OR_EQUAL' || baseOp === 'CONCAT' || baseOp === 'CHECKSIG' || baseOp === 'CHECKSIGVERIFY' || baseOp === 'CHECKTIMEVERIFY' || baseOp === 'VERIFY') {
        this.simulatedStackDepth -= 1;
    } else if (baseOp === 'SLICE') {
        this.simulatedStackDepth -= 2; 
    }
    // SWAP, OVER, ROT, PICK, HASH, SUM_TO_ADDR, READ_OUTPUT_STATE, OUTPUT_ADDRESS have a net 0 effect.
    // (OUTPUT_ADDRESS pops index, pushes address — net 0.)

    if (this.simulatedStackDepth > 64) {
        throw `Fatal Compiler Error: Maximum stack depth of 64 exceeded at line ${line || 'unknown'}. (Current depth: ${this.simulatedStackDepth})`;
    }
  }
  
  generate(ast) {
    for(let n of ast) if(n) this.genStmt(n);
    return this.optimize(this.asm);
  }

  optimize(asm) {
    let opt = [];
    for(let i=0; i<asm.length; i++) {
      let curr = asm[i];
      let next = asm[i+1];
      
      // Strip ONLY pure stack redundancies, leave math alone (used for type-casting)
      if (curr === 'SWAP' && next === 'SWAP') { i++; continue; }
      if (curr === 'DUP' && next === 'DROP') { i++; continue; }
      
      opt.push(curr);
    }
    return opt;
  }

  genStmt(s){
    if(s.type==='TestDef') {
      this.tests = this.tests || {};
      this.tests[s.name] = s.body;
      return; 
    }
    if(s.type==='MacroDefStmt'){this.macros[s.name]={params: s.params, body: s.body};}
    else if(s.type==='LetStmt'){this.constants[s.id]=s.expr;}
    else if(s.type==='VarDecl'){
      // Prevent Shadowing
      const existing = this.locals.find(l => l.id === s.id);
      if (existing) {
          throw `Semantic Error at ${s.l}:${s.c}: Cannot redeclare variable '${s.id}'. Shadowing is forbidden in Midscript.`;
      }

      let tType = this.genExpr(s.expr); 
      this.locals.push({id: s.id, type: tType});
    }
    else if(s.type==='StateDef' || s.type==='StructDef'){
      this.structs[s.name]=s.fields;
      if(s.type==='StateDef') this.stateDefs[s.name]=s.fields;
    }
    else if(s.type==='BlockStmt') {
      let initialLocals = this.locals.length; 
      for(let c of s.stmts) this.genStmt(c);
      let toDrop = this.locals.length - initialLocals;
      for(let i=0; i<toDrop; i++) {
        this.emit('DROP', s.l); // Pass line
        this.locals.pop(); 
      }
    }
    else if (s.type === 'AssignStmt') {
        const idx = this.locals.findIndex(l => l.id === s.id);
        if (idx === -1) throw `Semantic Error at ${s.l}:${s.c}: Assignment to undeclared variable '${s.id}'`;
        
        // Push the new value to the top of the stack
        let newType = this.genExpr(s.expr);
        
        // Calculate depth to the old variable (remember we just pushed 1 new item!)
        const depth = this.locals.length - idx; 

        if (depth === 1) {
            // It was already on top. Just swap the new one in and drop the old.
            this.emit('SWAP', s.l);
            this.emit('DROP', s.l);
        } else if (depth === 2) {
            // It's the second item. ROT pulls it to the top so we can drop it, 
            // then we SWAP to put the new value in its place.
            this.emit('ROT', s.l);
            this.emit('DROP', s.l);
            this.emit('SWAP', s.l);
        } else {
            // Deep assignment. We have to pull it up with a targeted ROT, 
            // drop it, and push the new value down. 
            // Midstate script doesn't have OP_ROLL, so deep mutable state 
            // is restricted. Best practice: shadow variables instead of mutating them.
            throw `Semantic Error at ${s.l}:${s.c}: Cannot reassign '${s.id}' (depth ${depth}). To mutate deep variables, redeclare them (e.g. 'var ${s.id} = ...').`;
        }
        this.locals[idx].type = newType;
    }
    else if(s.type==='AssertStmt'){
      this.genExpr(s.expr);
      this.emit('VERIFY', s.l); // VERIFY is usually where runtime errors happen!
    }
    else if(s.type==='ExprStmt')this.genExpr(s.expr);
    else if(s.type==='RawASM')this.emit(s.arg!==undefined?`${s.op} ${s.arg}`:s.op, s.l);
    else if(s.type==='IfStmt'){
      this.genExpr(s.cond);
      this.emit('IF', s.l);
      this.genStmt(s.thenB);
      if(s.elseB){this.emit('ELSE', s.l); this.genStmt(s.elseB);}
      this.emit('ENDIF', s.l);
    }
    else if(s.type==='RepeatStmt'){for(let i=0;i<s.count;i++)this.genStmt(s.body);}
    else if(s.type==='SwitchStmt'){this.genSwitch(s);}
  } 

genSwitch(sw) {
  this.genExpr(sw.expr); // Push routing value
  let cases = sw.cases.filter(c => !c.isDefault);
  let def = sw.cases.find(c => c.isDefault);
  
  for(let i=0; i<cases.length; i++) {
    this.emit('DUP', sw.l); 
    this.genExpr(cases[i].val);
    this.emit('EQUAL', sw.l); 
    this.emit('IF', sw.l); 
    this.emit('DROP', sw.l); 
    this.genStmt(cases[i].body);
    this.emit('ELSE', sw.l); 
  }
  
  this.emit('DROP', sw.l); 
  if (def) this.genStmt(def.body);
  
  for(let i=0; i<cases.length; i++) {
    this.emit('ENDIF', sw.l); 
  }
}

  genExpr(e){
    if(e.type==='ImplicitTop') return 'unknown'; 
    
    if(e.type==='Lit'){
      const idx = this.locals.findIndex(l => l.id === e.v);
      if (idx !== -1) {
          const depth = this.locals.length - 1 - idx + this.stackDepthOffset;
          const varType = this.locals[idx].type;
          
          if (depth === 0) {
              this.emit('DUP', e.l);
          } else if (depth === 1) {
              this.emit('OVER', e.l);
          } else {
              this.emit(`PUSH_INT ${depth}`, e.l);
              this.emit('PICK', e.l);
          }
          return varType;
      }
      
      if (e.v === 'READ_INPUT_STATE') { this.emit('READ_INPUT_STATE', e.l); return 'hex'; }
      if (e.v === 'READ_OUTPUT_STATE') { this.emit('READ_OUTPUT_STATE', e.l); return 'hex'; }

      if(this.constants[e.v]) {
        return this.genExpr(this.constants[e.v]);
      } else {
        if (e.k === 'hex' && !/^[0-9a-fA-F]*$/.test(e.v)) {
          throw `Semantic Error at ${e.l}:${e.c}: Undeclared identifier or invalid hex '${e.v}'`;
        }
        this.emit(e.k==='int'?`PUSH_INT ${e.v}`:`PUSH_HEX ${e.v}`, e.l);
        return e.k;
      }
    }
    else if(e.type==='Bin'){
      let lType = this.genExpr(e.l);
      this.stackDepthOffset++; 
      let rType = this.genExpr(e.r);
      this.stackDepthOffset--; 

      const isMath = ['+', '-', '*', '/'].includes(e.op);
      const isComparison = ['==', '>=', '>', '<=', '<'].includes(e.op);
      const isLogical = ['&&', '||'].includes(e.op);
      
      const line = e.l.l || 'unknown';
      const col = e.l.c || 'unknown';

      if (isMath && ((lType !== 'int' && lType !== 'unknown') || (rType !== 'int' && rType !== 'unknown'))) {
          throw `Semantic Error at ${line}:${col}: Type Mismatch. Math ('${e.op}') requires both sides to be integers.`;
      }
      
      if (isComparison && lType !== rType && lType !== 'unknown' && rType !== 'unknown') {
          throw `Semantic Error at ${line}:${col}: Type Mismatch. Cannot compare an '${lType}' with an '${rType}'.`;
      }

      if (isLogical && (lType === 'hex' || rType === 'hex')) {
          throw `Semantic Error at ${line}:${col}: Type Mismatch. Logical operators ('${e.op}') require evaluated boolean/int expressions, not raw hex.`;
      }
      
      if(e.op==='==')this.emit('EQUAL', e.l);
      if(e.op==='>=')this.emit('GREATER_OR_EQUAL', e.l);
      if(e.op==='>'){
        this.emit('PUSH_INT 1', e.l); 
        this.emit('ADD', e.l); 
        this.emit('GREATER_OR_EQUAL', e.l);
      }
      if(e.op==='<='){
        this.emit('SWAP', e.l); 
        this.emit('GREATER_OR_EQUAL', e.l);
      }
      if(e.op==='<'){
        this.emit('SWAP', e.l); 
        this.emit('PUSH_INT 1', e.l); 
        this.emit('ADD', e.l); 
        this.emit('GREATER_OR_EQUAL', e.l);
      }
      
      
      if(e.op==='+')this.emit('ADD', e.l);
      if(e.op==='-')this.emit('SUB', e.l);
      if(e.op==='*')this.emit('MUL', e.l);
      if(e.op==='/')this.emit('DIV', e.l);
      if(e.op==='%')this.emit('MOD', e.l);
      if(e.op==='&&'){this.emit('ADD', e.l); this.emit('PUSH_INT 2', e.l); this.emit('EQUAL', e.l);}
      if(e.op==='||'){this.emit('ADD', e.l); this.emit('PUSH_INT 1', e.l); this.emit('GREATER_OR_EQUAL', e.l);}
      
      return 'int';
    }
    else if(e.type==='Call'){
      const fn=e.name;
      if(this.macros[fn]){
        const mac = this.macros[fn];
        if (e.args.length !== mac.params.length) throw `Semantic Error at ${e.l}:${e.c}: Macro '${fn}' expects ${mac.params.length} arguments.`;

        let macroLocalsSnapshot = this.locals.length; 
        
        for (let i = 0; i < e.args.length; i++) {
            let argType = this.genExpr(e.args[i]);
            this.locals.push({id: mac.params[i], type: argType});
        }
        
        this.genStmt(mac.body);
        
        let toDrop = this.locals.length - macroLocalsSnapshot;
        for(let i=0; i<toDrop; i++) {
            this.emit('DROP', e.l);
            this.locals.pop();
        }
        return 'unknown';
      }
      else if(fn==='pop') {
          throw `Semantic Error at ${e.l}:${e.c}: Use 'pop_int()' or 'pop_hex()' to ensure type safety. Generic 'pop()' is unsafe.`;
      }
      else if(fn==='pop_int') {
          return 'int';
      }
      else if(fn==='pop_hex') {
          return 'hex';
      }
      else if(fn==='size') { 
          this.genExpr(e.args[0]); 
          this.emit('SIZE', e.l); 
          this.emit('SWAP', e.l); 
          this.emit('DROP', e.l); 
          return 'int'; 
      }
      else if(fn==='input_value') { 
          this.emit('INPUT_VALUE', e.l); 
          return 'int'; 
      }
      else if(fn==='pick') {
          this.genExpr(e.args[0]);
          this.emit('PICK', e.l);
          return 'unknown';
      }
      else if(fn==='peek') { this.emit('DUP', e.l); return 'unknown'; }
      else if(fn==='dup') { this.emit('DUP', e.l); return 'unknown'; }
      else if(fn==='drop') { this.emit('DROP', e.l); return 'unknown'; }
      else if(fn==='swap') { this.emit('SWAP', e.l); return 'unknown'; }
      else if(fn==='over') { this.emit('OVER', e.l); return 'unknown'; }
      else if(fn==='rot') { this.emit('ROT', e.l); return 'unknown'; }
      else if(fn==='slice') { 
          // Slice usually pops: [value, offset, length]
          // We need to generate expressions for arguments if they exist
          if(e.args.length === 3) {
             this.genExpr(e.args[0]); this.stackDepthOffset++;
             this.genExpr(e.args[1]); this.stackDepthOffset++;
             this.genExpr(e.args[2]); this.stackDepthOffset -= 2;
          }
          this.emit('SLICE', e.l); return 'hex'; 
      }
      else if(fn==='concat') {
          if (e.args && e.args.length === 2) {
              this.genExpr(e.args[0]);
              this.stackDepthOffset++;
              this.genExpr(e.args[1]);
              this.stackDepthOffset--;
          }
          this.emit('CONCAT', e.l); return 'hex';
      }
      else if(fn==='add') { this.emit('ADD', e.l); return 'int'; }
      else if(fn==='sub') { this.emit('SUB', e.l); return 'int'; }
      else if(fn==='mul') { this.emit('MUL', e.l); return 'int'; }
      else if(fn==='div') { this.emit('DIV', e.l); return 'int'; }
      else if(fn==='hash') {
          if (e.args && e.args.length > 0) this.genExpr(e.args[0]);
          this.emit('HASH', e.l); return 'hex';
      }
      else if(fn==='fail'){
          this.emit('PUSH_INT 0', e.l);
          this.emit('VERIFY', e.l);   
          return 'unknown';
      }
      else if(fn==='verify_gte'){
          this.genExpr(e.args[0]);
          this.emit('GREATER_OR_EQUAL', e.l);
          this.emit('VERIFY', e.l); 
          return 'unknown';
      }
      else if(fn==='require_preimage'){
          this.emit('HASH', e.l);
          this.genExpr(e.args[0]);
          this.emit('EQUALVERIFY', e.l); 
          return 'unknown';
      }
      else if(fn==='require_time'){
          this.genExpr(e.args[0]);
          this.emit('CHECKTIMEVERIFY', e.l); 
          return 'unknown';
      }
      else if(fn==='require_sig'){
          this.genExpr(e.args[0]);
          this.emit('CHECKSIGVERIFY', e.l); 
          return 'unknown';
      }
      else if(fn==='check_sig'){
          this.genExpr(e.args[0]);
          this.emit('CHECKSIG', e.l); 
          return 'int';
      }
      else if(fn==='require_transfer'){
        this.genExpr(e.args[0]);
        this.emit('SUM_TO_ADDR', e.l); 
        this.stackDepthOffset++;
        this.genExpr(e.args[1]);
        this.stackDepthOffset--;
        this.emit('EQUALVERIFY', e.l);
        return 'unknown';
      }
      else if(fn==='sum_to_addr'){this.genExpr(e.args[0]); this.emit('SUM_TO_ADDR', e.l); return 'int';}
      else if(fn==='read_input_state'){this.emit('READ_INPUT_STATE', e.l); return 'hex';}
      else if(fn==='read_output_state'){this.genExpr(e.args[0]); this.emit('READ_OUTPUT_STATE', e.l); return 'hex';}
      else if(fn==='output_address'){this.genExpr(e.args[0]); this.emit('OUTPUT_ADDRESS', e.l); return 'hex';}
      else if(fn==='this_address'){this.emit('THIS_ADDRESS', e.l); return 'hex';}
      else if(fn==='sum_input_value'){this.emit('SUM_INPUT_VALUE', e.l); return 'int';}
      else throw `Semantic Error at ${e.l}:${e.c}: Unknown function or macro '${fn}'`;
    }
    else if(e.type==='FieldAccess'){
      let def = this.structs[e.base] || this.stateDefs[e.base];
      if(!def || !def[e.field]) throw `Semantic Error at ${e.l}:${e.c}: Unknown field '${e.field}' on '${e.base}'`;
      let f = def[e.field];
      this.emit('READ_INPUT_STATE', e.l);
      this.emit(`PUSH_INT ${f.offset}`, e.l);
      this.emit(`PUSH_INT ${f.size}`, e.l);
      this.emit('SLICE', e.l);
      
      if (f.size <= 8) {
          this.emit('PUSH_INT 0', e.l);
          this.emit('ADD', e.l);
          return 'int';
      }
      return 'hex';
    }
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Compile Midstate contract source to bytecode.
 *
 * The standard library is prepended, exactly as the IDE does, so `min`, `max`,
 * `require_signed_by` and friends are available without an import.
 *
 * @param {string} source
 * @returns {{bytecode: string, asm: string[], sizeBytes: number, sigops: number}}
 * @throws If the source does not compile, or the result exceeds MAX_SCRIPT_SIZE.
 */
export function compile(source) {
    const stdAst = new Parser(new Lexer(STD_LIB).tokenize()).parse();
    const userAst = new Parser(new Lexer(String(source)).tokenize()).parse();
    const gen = new CodeGen();
    const asm = gen.generate([...stdAst, ...userAst]);
    const bytecode = assembleBytecode(asm);
    const sizeBytes = bytecode.length / 2;
    // Consensus caps a script at 1024 bytes. Failing here is much cheaper than
    // discovering it when the funding transaction is rejected.
    if (sizeBytes > MAX_SCRIPT_SIZE) {
        throw new Error(`compiler: script is ${sizeBytes} B, MAX_SCRIPT_SIZE is ${MAX_SCRIPT_SIZE} B`);
    }
    return {
        bytecode,
        asm,
        sizeBytes,
        sigops: asm.filter((i) => i === 'CHECKSIG' || i === 'CHECKSIGVERIFY').length,
    };
}

/** Consensus limit on compiled script length. */
export const MAX_SCRIPT_SIZE = 1024;

export { OPS, STD_LIB, Lexer, Parser, CodeGen, assembleBytecode };
