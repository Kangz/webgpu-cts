// pass v-0033: variable 'a' and its initializer have the same storetype, 'i32'.

var<out> a : i32  = 0;

[[stage(vertex)]]
fn main() -> void {
}
