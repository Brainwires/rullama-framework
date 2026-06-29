// BenchApp: minimal headless iOS app that calls into the rullama-ios-bench
// static library and prints timing results to stdout (captured via
// idevicesyslog from the Mac).
//
// We deliberately don't construct a UIWindow / scene — iOS will run the
// `main()` function, our Rust bench will execute synchronously, results
// flush to stdout, and then we explicitly exit. This is a "tool" target
// rather than a polished app.

import Foundation
import UIKit

// The Rust crate exports `rullama_run_bench` and `rullama_describe_adapter`
// with C ABI.  Linker sees them; we just declare the signatures.
@_silgen_name("rullama_run_bench")
func rullama_run_bench() -> Int32

@_silgen_name("rullama_describe_adapter")
func rullama_describe_adapter() -> UnsafePointer<CChar>

// Force unbuffered stdout so syslog sees output as it's produced.
setbuf(stdout, nil)
setbuf(stderr, nil)

print("BenchApp: starting…")
let info = String(cString: rullama_describe_adapter())
print("BenchApp: lib=\(info)")
let rc = rullama_run_bench()
print("BenchApp: rullama_run_bench returned \(rc)")
print("BenchApp: exiting")
exit(Int32(rc))
