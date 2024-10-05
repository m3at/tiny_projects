!function(e,n){"object"==typeof exports&&"undefined"!=typeof module?n(exports):"function"==typeof define&&define.amd?define(["exports"],n):n((e="undefined"!=typeof globalThis?globalThis:e||self).Diff={})}(this,function(e){"use strict";function r(){}function w(e,n,t,r,i){for(var o,l=[];n;)l.push(n),o=n.previousComponent,delete n.previousComponent,n=o;l.reverse();for(var a=0,u=l.length,s=0,f=0;a<u;a++){var c,d=l[a];d.removed?(d.value=e.join(r.slice(f,f+d.count)),f+=d.count):(!d.added&&i?(c=(c=t.slice(s,s+d.count)).map(function(e,n){n=r[f+n];return n.length>e.length?n:e}),d.value=e.join(c)):d.value=e.join(t.slice(s,s+d.count)),s+=d.count,d.added||(f+=d.count))}return l}r.prototype={diff:function(l,a){var u=2<arguments.length&&void 0!==arguments[2]?arguments[2]:{},n=u.callback,s=("function"==typeof u&&(n=u,u={}),this);function f(e){return e=s.postProcess(e,u),n?(setTimeout(function(){n(e)},0),!0):e}l=this.castInput(l,u),a=this.castInput(a,u),l=this.removeEmpty(this.tokenize(l,u));var c=(a=this.removeEmpty(this.tokenize(a,u))).length,d=l.length,h=1,t=c+d,e=(null!=u.maxEditLength&&(t=Math.min(t,u.maxEditLength)),null!=(e=u.timeout)?e:1/0),r=Date.now()+e,p=[{oldPos:-1,lastComponent:void 0}],v=this.extractCommon(p[0],a,l,0,u);if(p[0].oldPos+1>=d&&c<=v+1)return f(w(s,p[0].lastComponent,a,l,s.useLongestToken));var g=-1/0,m=1/0;function i(){for(var e=Math.max(g,-h);e<=Math.min(m,h);e+=2){var n=void 0,t=p[e-1],r=p[e+1],i=(t&&(p[e-1]=void 0),!1),o=(r&&(o=r.oldPos-e,i=r&&0<=o&&o<c),t&&t.oldPos+1<d);if(i||o){if(n=!o||i&&t.oldPos<r.oldPos?s.addToPath(r,!0,!1,0,u):s.addToPath(t,!1,!0,1,u),v=s.extractCommon(n,a,l,e,u),n.oldPos+1>=d&&c<=v+1)return f(w(s,n.lastComponent,a,l,s.useLongestToken));(p[e]=n).oldPos+1>=d&&(m=Math.min(m,e-1)),c<=v+1&&(g=Math.max(g,e+1))}else p[e]=void 0}h++}if(n)!function e(){setTimeout(function(){if(t<h||Date.now()>r)return n();i()||e()},0)}();else for(;h<=t&&Date.now()<=r;){var o=i();if(o)return o}},addToPath:function(e,n,t,r,i){var o=e.lastComponent;return o&&!i.oneChangePerToken&&o.added===n&&o.removed===t?{oldPos:e.oldPos+r,lastComponent:{count:o.count+1,added:n,removed:t,previousComponent:o.previousComponent}}:{oldPos:e.oldPos+r,lastComponent:{count:1,added:n,removed:t,previousComponent:o}}},extractCommon:function(e,n,t,r,i){for(var o=n.length,l=t.length,a=e.oldPos,u=a-r,s=0;u+1<o&&a+1<l&&this.equals(t[a+1],n[u+1],i);)u++,a++,s++,i.oneChangePerToken&&(e.lastComponent={count:1,previousComponent:e.lastComponent,added:!1,removed:!1});return s&&!i.oneChangePerToken&&(e.lastComponent={count:s,previousComponent:e.lastComponent,added:!1,removed:!1}),e.oldPos=a,u},equals:function(e,n,t){return t.comparator?t.comparator(e,n):e===n||t.ignoreCase&&e.toLowerCase()===n.toLowerCase()},removeEmpty:function(e){for(var n=[],t=0;t<e.length;t++)e[t]&&n.push(e[t]);return n},castInput:function(e){return e},tokenize:function(e){return Array.from(e)},join:function(e){return e.join("")},postProcess:function(e){return e}};var I=new r;function u(e,n){for(var t=0;t<e.length&&t<n.length;t++)if(e[t]!=n[t])return e.slice(0,t);return e.slice(0,t)}function s(e,n){var t;if(!e||!n||e[e.length-1]!=n[n.length-1])return"";for(t=0;t<e.length&&t<n.length;t++)if(e[e.length-(t+1)]!=n[n.length-(t+1)])return e.slice(-t);return e.slice(-t)}function f(e,n,t){if(e.slice(0,n.length)!=n)throw Error("string ".concat(JSON.stringify(e)," doesn't start with prefix ").concat(JSON.stringify(n),"; this is a bug"));return t+e.slice(n.length)}function c(e,n,t){if(!n)return e+t;if(e.slice(-n.length)!=n)throw Error("string ".concat(JSON.stringify(e)," doesn't end with suffix ").concat(JSON.stringify(n),"; this is a bug"));return e.slice(0,-n.length)+t}function d(e,n){return f(e,n,"")}function h(e,n){return c(e,n,"")}function p(e,n){return n.slice(0,function(e,n){var t=0;e.length>n.length&&(t=e.length-n.length);var r=n.length;e.length<n.length&&(r=e.length);var i=Array(r),o=0;i[0]=0;for(var l=1;l<r;l++){for(n[l]==n[o]?i[l]=i[o]:i[l]=o;0<o&&n[l]!=n[o];)o=i[o];n[l]==n[o]&&o++}o=0;for(var a=t;a<e.length;a++){for(;0<o&&e[a]!=n[o];)o=i[o];e[a]==n[o]&&o++}return o}(e,n))}var t="a-zA-Z\\u{C0}-\\u{FF}\\u{D8}-\\u{F6}\\u{F8}-\\u{2C6}\\u{2C8}-\\u{2D7}\\u{2DE}-\\u{2FF}\\u{1E00}-\\u{1EFF}",z=new RegExp("[".concat(t,"]+|\\s+|[^").concat(t,"]"),"ug"),i=new r;function o(e,n,t,r){var i,o,l,a;n&&t?(i=n.value.match(/^\s*/)[0],o=n.value.match(/\s*$/)[0],l=t.value.match(/^\s*/)[0],a=t.value.match(/\s*$/)[0],e&&(i=u(i,l),e.value=c(e.value,l,i),n.value=d(n.value,i),t.value=d(t.value,i)),r&&(l=s(o,a),r.value=f(r.value,a,l),n.value=h(n.value,l),t.value=h(t.value,l))):t?(e&&(t.value=t.value.replace(/^\s*/,"")),r&&(r.value=r.value.replace(/^\s*/,""))):e&&r?(i=r.value.match(/^\s*/)[0],o=n.value.match(/^\s*/)[0],a=n.value.match(/\s*$/)[0],l=u(i,o),n.value=d(n.value,l),t=s(d(i,l),a),n.value=h(n.value,t),r.value=f(r.value,i,t),e.value=c(e.value,i,i.slice(0,i.length-t.length))):r?(o=r.value.match(/^\s*/)[0],l=p(n.value.match(/\s*$/)[0],o),n.value=h(n.value,l)):e&&(a=p(e.value.match(/\s*$/)[0],n.value.match(/^\s*/)[0]),n.value=d(n.value,a))}i.equals=function(e,n,t){return t.ignoreCase&&(e=e.toLowerCase(),n=n.toLowerCase()),e.trim()===n.trim()},i.tokenize=function(e){var n,t=1<arguments.length&&void 0!==arguments[1]?arguments[1]:{};if(t.intlSegmenter){if("word"!=t.intlSegmenter.resolvedOptions().granularity)throw new Error('The segmenter passed must have a granularity of "word"');n=Array.from(t.intlSegmenter.segment(e),function(e){return e.segment})}else n=e.match(z)||[];var r=[],i=null;return n.forEach(function(e){/\s/.test(e)?r.push(null==i?e:r.pop()+e):/\s/.test(i)?r.push(r[r.length-1]==i?r.pop()+e:i+e):r.push(e),i=e}),r},i.join=function(e){return e.map(function(e,n){return 0==n?e:e.replace(/^\s+/,"")}).join("")},i.postProcess=function(e,n){var t,r,i;return e&&!n.oneChangePerToken&&(i=r=t=null,e.forEach(function(e){e.added?r=e:i=e.removed?e:((r||i)&&o(t,i,r,e),t=e,r=null)}),r||i)&&o(t,i,r,null),e};var l=new r;function a(e,n,t){return l.diff(e,n,t)}l.tokenize=function(e){var n=new RegExp("(\\r?\\n)|[".concat(t,"]+|[^\\S\\n\\r]+|[^").concat(t,"]"),"ug");return e.match(n)||[]};var v=new r;function y(e,n,t){return v.diff(e,n,t)}v.tokenize=function(e,n){var t=[],r=(e=n.stripTrailingCr?e.replace(/\r\n/g,"\n"):e).split(/(\n|\r\n)/);r[r.length-1]||r.pop();for(var i=0;i<r.length;i++){var o=r[i];i%2&&!n.newlineIsToken?t[t.length-1]+=o:t.push(o)}return t},v.equals=function(e,n,t){return t.ignoreWhitespace?(t.newlineIsToken&&e.includes("\n")||(e=e.trim()),t.newlineIsToken&&n.includes("\n")||(n=n.trim())):t.ignoreNewlineAtEof&&!t.newlineIsToken&&(e.endsWith("\n")&&(e=e.slice(0,-1)),n.endsWith("\n"))&&(n=n.slice(0,-1)),r.prototype.equals.call(this,e,n,t)};var g=new r;g.tokenize=function(e){return e.split(/(\S.+?[.!?])(?=\s+|$)/)};var m=new r;function n(n,e){var t,r=Object.keys(n);return Object.getOwnPropertySymbols&&(t=Object.getOwnPropertySymbols(n),e&&(t=t.filter(function(e){return Object.getOwnPropertyDescriptor(n,e).enumerable})),r.push.apply(r,t)),r}function b(r){for(var e=1;e<arguments.length;e++){var i=null!=arguments[e]?arguments[e]:{};e%2?n(Object(i),!0).forEach(function(e){var n,t;n=r,t=i[e=e],(e=function(e){e=function(e,n){if("object"!=typeof e||!e)return e;var t=e[Symbol.toPrimitive];if(void 0===t)return("string"===n?String:Number)(e);t=t.call(e,n||"default");if("object"!=typeof t)return t;throw new TypeError("@@toPrimitive must return a primitive value.")}(e,"string");return"symbol"==typeof e?e:e+""}(e))in n?Object.defineProperty(n,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):n[e]=t}):Object.getOwnPropertyDescriptors?Object.defineProperties(r,Object.getOwnPropertyDescriptors(i)):n(Object(i)).forEach(function(e){Object.defineProperty(r,e,Object.getOwnPropertyDescriptor(i,e))})}return r}function L(e){return(L="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function k(e){return function(e){if(Array.isArray(e))return S(e)}(e)||function(e){if("undefined"!=typeof Symbol&&null!=e[Symbol.iterator]||null!=e["@@iterator"])return Array.from(e)}(e)||function(e,n){var t;if(e)return"string"==typeof e?S(e,n):"Map"===(t="Object"===(t=Object.prototype.toString.call(e).slice(8,-1))&&e.constructor?e.constructor.name:t)||"Set"===t?Array.from(e):"Arguments"===t||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t)?S(e,n):void 0}(e)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function S(e,n){(null==n||n>e.length)&&(n=e.length);for(var t=0,r=new Array(n);t<n;t++)r[t]=e[t];return r}m.tokenize=function(e){return e.split(/([{}:;,]|\s+)/)};var x=new r;function P(e,n,t,r,i){var o,l;for(n=n||[],t=t||[],r&&(e=r(i,e)),o=0;o<n.length;o+=1)if(n[o]===e)return t[o];if("[object Array]"===Object.prototype.toString.call(e)){for(n.push(e),l=new Array(e.length),t.push(l),o=0;o<e.length;o+=1)l[o]=P(e[o],n,t,r,i);n.pop(),t.pop()}else if("object"===L(e=e&&e.toJSON?e.toJSON():e)&&null!==e){n.push(e),t.push(l={});var a,u=[];for(a in e)Object.prototype.hasOwnProperty.call(e,a)&&u.push(a);for(u.sort(),o=0;o<u.length;o+=1)l[a=u[o]]=P(e[a],n,t,r,a);n.pop(),t.pop()}else l=e;return l}x.useLongestToken=!0,x.tokenize=v.tokenize,x.castInput=function(e,n){var t=n.undefinedReplacement,n=n.stringifyReplacer,n=void 0===n?function(e,n){return void 0===n?t:n}:n;return"string"==typeof e?e:JSON.stringify(P(e,null,null,n),n,"  ")},x.equals=function(e,n,t){return r.prototype.equals.call(x,e.replace(/,([\r\n])/g,"$1"),n.replace(/,([\r\n])/g,"$1"),t)};var F=new r;function N(e){return Array.isArray(e)?e.map(N):b(b({},e),{},{hunks:e.hunks.map(function(t){return b(b({},t),{},{lines:t.lines.map(function(e,n){return e.startsWith("\\")||e.endsWith("\r")||null!=(n=t.lines[n+1])&&n.startsWith("\\")?e:e+"\r"})})})})}function C(e){return Array.isArray(e)?e.map(C):b(b({},e),{},{hunks:e.hunks.map(function(e){return b(b({},e),{},{lines:e.lines.map(function(e){return e.endsWith("\r")?e.substring(0,e.length-1):e})})})})}function j(e){var l=e.split(/\n/),r=[],a=0;function n(){var e={};for(r.push(e);a<l.length;){var n=l[a];if(/^(\-\-\-|\+\+\+|@@)\s/.test(n))break;n=/^(?:Index:|diff(?: -r \w+)+)\s+(.+?)\s*$/.exec(n);n&&(e.index=n[1]),a++}for(i(e),i(e),e.hunks=[];a<l.length;){var t=l[a];if(/^(Index:\s|diff\s|\-\-\-\s|\+\+\+\s|===================================================================)/.test(t))break;if(/^@@/.test(t))e.hunks.push(function(){var e=a,n=l[a++].split(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/),t={oldStart:+n[1],oldLines:void 0===n[2]?1:+n[2],newStart:+n[3],newLines:void 0===n[4]?1:+n[4],lines:[]};0===t.oldLines&&(t.oldStart+=1);0===t.newLines&&(t.newStart+=1);for(var r=0,i=0;a<l.length&&(i<t.oldLines||r<t.newLines||null!=(o=l[a])&&o.startsWith("\\"));a++){var o=0==l[a].length&&a!=l.length-1?" ":l[a][0];if("+"!==o&&"-"!==o&&" "!==o&&"\\"!==o)throw new Error("Hunk at line ".concat(e+1," contained invalid line ").concat(l[a]));t.lines.push(l[a]),"+"===o?r++:"-"===o?i++:" "===o&&(r++,i++)}r||1!==t.newLines||(t.newLines=0);i||1!==t.oldLines||(t.oldLines=0);if(r!==t.newLines)throw new Error("Added line count did not match for hunk at line "+(e+1));if(i===t.oldLines)return t;throw new Error("Removed line count did not match for hunk at line "+(e+1))}());else{if(t)throw new Error("Unknown line "+(a+1)+" "+JSON.stringify(t));a++}}}function i(e){var n,t,r=/^(---|\+\+\+)\s+(.*)\r?$/.exec(l[a]);r&&(n="---"===r[1]?"old":"new",t=(r=r[2].split("\t",2))[0].replace(/\\\\/g,"\\"),/^".*"$/.test(t)&&(t=t.substr(1,t.length-2)),e[n+"FileName"]=t,e[n+"Header"]=(r[1]||"").trim(),a++)}for(;a<l.length;)n();return r}function D(e,n){var t,r=2<arguments.length&&void 0!==arguments[2]?arguments[2]:{};if("string"==typeof n&&(n=j(n)),Array.isArray(n)){if(1<n.length)throw new Error("applyPatch only works with a single input.");n=n[0]}!r.autoConvertLineEndings&&null!=r.autoConvertLineEndings||(!(t=e).includes("\r\n")||t.match(/(?<!\r)\n/)||(t=n,(t=Array.isArray(t)?t:[t]).some(function(e){return e.hunks.some(function(e){return e.lines.some(function(e){return!e.startsWith("\\")&&e.endsWith("\r")})})}))?!(t=e).includes("\r\n")&&t.includes("\n")&&(t=n,(t=Array.isArray(t)?t:[t]).some(function(e){return e.hunks.some(function(e){return e.lines.some(function(e){return e.endsWith("\r")})})}))&&t.every(function(e){return e.hunks.every(function(t){return t.lines.every(function(e,n){return e.startsWith("\\")||e.endsWith("\r")||(null==(e=t.lines[n+1])?void 0:e.startsWith("\\"))})})})&&(n=C(n)):n=N(n));var g=e.split("\n"),i=n.hunks,m=r.compareLine||function(e,n,t,r){return n===r},o=r.fuzzFactor||0,l=0;if(o<0||!Number.isInteger(o))throw new Error("fuzzFactor must be a non-negative integer");if(!i.length)return e;for(var a="",u=!1,s=!1,f=0;f<i[i.length-1].lines.length;f++){var c=i[i.length-1].lines[f];"\\"==c[0]&&("+"==a[0]?u=!0:"-"==a[0]&&(s=!0)),a=c}if(u){if(s){if(!o&&""==g[g.length-1])return!1}else if(""==g[g.length-1])g.pop();else if(!o)return!1}else if(s)if(""!=g[g.length-1])g.push("");else if(!o)return!1;for(var d=[],h=0,p=0;p<i.length;p++){for(var v=i[p],w=void 0,y=g.length-v.oldLines+o,b=void 0,L=0;L<=o;L++){for(var k=function(n,t,r){var i=!0,o=!1,l=!1,a=1;return function e(){if(i&&!l){if(o?a++:i=!1,n+a<=r)return n+a;l=!0}if(!o)return l||(i=!0),t<=n-a?n-a++:(o=!0,e())}}(b=v.oldStart+h-1,l,y);void 0!==b&&!(w=function e(n,t,r,i,o,l,a){for(var u=3<arguments.length&&void 0!==i?i:0,s=!(4<arguments.length&&void 0!==o)||o,f=5<arguments.length&&void 0!==l?l:[],c=6<arguments.length&&void 0!==a?a:0,d=0,h=!1;u<n.length;u++){var p=0<(v=n[u]).length?v[0]:" ",v=0<v.length?v.substr(1):v;if("-"===p){if(!m(t+1,g[t],p,v))return r&&null!=g[t]?(f[c]=g[t],e(n,t+1,r-1,u,!1,f,c+1)):null;t++,d=0}if("+"===p){if(!s)return null;f[c]=v,c++,h=!(d=0)}if(" "===p){if(d++,f[c]=g[t],!m(t+1,g[t],p,v))return h||!r?null:g[t]&&(e(n,t+1,r-1,u+1,!1,f,c+1)||e(n,t+1,r-1,u,!1,f,c+1))||e(n,t,r-1,u+1,!1,f,c);c++,h=!(s=!0),t++}}return t-=d,f.length=c-=d,{patchedLines:f,oldLineLastI:t-1}}(v.lines,b,L));b=k());if(w)break}if(!w)return!1;for(var S=l;S<b;S++)d.push(g[S]);for(var x=0;x<w.patchedLines.length;x++){var P=w.patchedLines[x];d.push(P)}l=w.oldLineLastI+1,h=b+1-v.oldStart}for(var F=l;F<g.length;F++)d.push(g[F]);return d.join("\n")}function O(p,v,e,n,g,m,w){if(void 0===(w="function"==typeof(w=w||{})?{callback:w}:w).context&&(w.context=4),w.newlineIsToken)throw new Error("newlineIsToken may not be used with patch-generation functions, only with diffing functions");if(!w.callback)return r(y(e,n,w));var t=w.callback;function r(r){if(r){r.push({value:"",lines:[]});for(var i=[],o=0,l=0,a=[],u=1,s=1,e=function(){var e,n=r[f],t=n.lines||function(e){var n=e.endsWith("\n"),e=e.split("\n").map(function(e){return e+"\n"});n?e.pop():e.push(e.pop().slice(0,-1));return e}(n.value);n.lines=t,n.added||n.removed?(o||(e=r[f-1],o=u,l=s,e&&(a=0<w.context?h(e.lines.slice(-w.context)):[],o-=a.length,l-=a.length)),a.push.apply(a,k(t.map(function(e){return(n.added?"+":"-")+e}))),n.added?s+=t.length:u+=t.length):(o&&(t.length<=2*w.context&&f<r.length-2?a.push.apply(a,k(h(t))):(e=Math.min(t.length,w.context),a.push.apply(a,k(h(t.slice(0,e)))),e={oldStart:o,oldLines:u-o+e,newStart:l,newLines:s-l+e,lines:a},i.push(e),l=o=0,a=[])),u+=t.length,s+=t.length)},f=0;f<r.length;f++)e();for(var n=0,t=i;n<t.length;n++)for(var c=t[n],d=0;d<c.lines.length;d++)c.lines[d].endsWith("\n")?c.lines[d]=c.lines[d].slice(0,-1):(c.lines.splice(d+1,0,"\\ No newline at end of file"),d++);return{oldFileName:p,newFileName:v,oldHeader:g,newHeader:m,hunks:i}}function h(e){return e.map(function(e){return" "+e})}}y(e,n,b(b({},w),{},{callback:function(e){e=r(e);t(e)}}))}function E(e){if(Array.isArray(e))return e.map(E).join("\n");var n=[];e.oldFileName==e.newFileName&&n.push("Index: "+e.oldFileName),n.push("==================================================================="),n.push("--- "+e.oldFileName+(void 0===e.oldHeader?"":"\t"+e.oldHeader)),n.push("+++ "+e.newFileName+(void 0===e.newHeader?"":"\t"+e.newHeader));for(var t=0;t<e.hunks.length;t++){var r=e.hunks[t];0===r.oldLines&&--r.oldStart,0===r.newLines&&--r.newStart,n.push("@@ -"+r.oldStart+","+r.oldLines+" +"+r.newStart+","+r.newLines+" @@"),n.push.apply(n,r.lines)}return n.join("\n")+"\n"}function M(e,n,t,r,i,o,l){if(null!=(u=l="function"==typeof l?{callback:l}:l)&&u.callback){var a=l.callback;O(e,n,t,r,i,o,b(b({},l),{},{callback:function(e){e?a(E(e)):a()}}))}else{var u=O(e,n,t,r,i,o,l);if(u)return E(u)}}function A(e,n){if(n.length>e.length)return!1;for(var t=0;t<n.length;t++)if(n[t]!==e[t])return!1;return!0}function $(e){var n=function r(e){var i=0;var o=0;e.forEach(function(e){var n,t;"string"!=typeof e?(n=r(e.mine),t=r(e.theirs),void 0!==i&&(n.oldLines===t.oldLines?i+=n.oldLines:i=void 0),void 0!==o&&(n.newLines===t.newLines?o+=n.newLines:o=void 0)):(void 0===o||"+"!==e[0]&&" "!==e[0]||o++,void 0===i||"-"!==e[0]&&" "!==e[0]||i++)});return{oldLines:i,newLines:o}}(e.lines),t=n.oldLines,n=n.newLines;void 0!==t?e.oldLines=t:delete e.oldLines,void 0!==n?e.newLines=n:delete e.newLines}function J(e,n){if("string"!=typeof e)return e;if(/^@@/m.test(e)||/^Index:/m.test(e))return j(e)[0];if(n)return O(void 0,void 0,n,e);throw new Error("Must provide a base reference or pass in a patch")}function q(e){return e.newFileName&&e.newFileName!==e.oldFileName}function H(e,n,t){return n===t?n:(e.conflict=!0,{mine:n,theirs:t})}function R(e,n){return e.oldStart<n.oldStart&&e.oldStart+e.oldLines<n.oldStart}function U(e,n){return{oldStart:e.oldStart,oldLines:e.oldLines,newStart:e.newStart+n,newLines:e.newLines,lines:e.lines}}function X(e,n,t,r){var i,n=W(n),t=function(e,n){var t=[],r=[],i=0,o=!1,l=!1;for(;i<n.length&&e.index<e.lines.length;){var a=e.lines[e.index],u=n[i];if("+"===u[0])break;if(o=o||" "!==a[0],r.push(u),i++,"+"===a[0])for(l=!0;"+"===a[0];)t.push(a),a=e.lines[++e.index];u.substr(1)===a.substr(1)?(t.push(a),e.index++):l=!0}"+"===(n[i]||"")[0]&&o&&(l=!0);if(l)return t;for(;i<n.length;)r.push(n[i++]);return{merged:r,changes:t}}(t,n);t.merged?(i=e.lines).push.apply(i,k(t.merged)):T(e,r?t:n,r?n:t)}function T(e,n,t){e.conflict=!0,e.lines.push({conflict:!0,mine:n,theirs:t})}function Z(e,n,t){for(;n.offset<t.offset&&n.index<n.lines.length;){var r=n.lines[n.index++];e.lines.push(r),n.offset++}}function B(e,n){for(;n.index<n.lines.length;){var t=n.lines[n.index++];e.lines.push(t)}}function W(e){for(var n=[],t=e.lines[e.index][0];e.index<e.lines.length;){var r=e.lines[e.index];if((t="-"===t&&"+"===r[0]?"+":t)!==r[0])break;n.push(r),e.index++}return n}function G(e){return e.reduce(function(e,n){return e&&"-"===n[0]},!0)}function K(e,n,t){for(var r=0;r<t;r++){var i=n[n.length-t+r].substr(1);if(e.lines[e.index+r]!==" "+i)return}return e.index+=t,1}F.tokenize=function(e){return e.slice()},F.join=F.removeEmpty=function(e){return e},e.Diff=r,e.applyPatch=D,e.applyPatches=function(e,i){"string"==typeof e&&(e=j(e));var n=0;!function t(){var r=e[n++];if(!r)return i.complete();i.loadFile(r,function(e,n){if(e)return i.complete(e);e=D(n,r,i),i.patched(r,e,function(e){if(e)return i.complete(e);t()})})}()},e.canonicalize=P,e.convertChangesToDMP=function(e){for(var n,t,r=[],i=0;i<e.length;i++)t=(n=e[i]).added?1:n.removed?-1:0,r.push([t,n.value]);return r},e.convertChangesToXML=function(e){for(var n=[],t=0;t<e.length;t++){var r=e[t];r.added?n.push("<ins>"):r.removed&&n.push("<del>"),n.push(r.value.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")),r.added?n.push("</ins>"):r.removed&&n.push("</del>")}return n.join("")},e.createPatch=function(e,n,t,r,i,o){return M(e,e,n,t,r,i,o)},e.createTwoFilesPatch=M,e.diffArrays=function(e,n,t){return F.diff(e,n,t)},e.diffChars=function(e,n,t){return I.diff(e,n,t)},e.diffCss=function(e,n,t){return m.diff(e,n,t)},e.diffJson=function(e,n,t){return x.diff(e,n,t)},e.diffLines=y,e.diffSentences=function(e,n,t){return g.diff(e,n,t)},e.diffTrimmedLines=function(e,n,t){return t=function(e,n){if("function"==typeof e)n.callback=e;else if(e)for(var t in e)e.hasOwnProperty(t)&&(n[t]=e[t]);return n}(t,{ignoreWhitespace:!0}),v.diff(e,n,t)},e.diffWords=function(e,n,t){return null==(null==t?void 0:t.ignoreWhitespace)||t.ignoreWhitespace?i.diff(e,n,t):a(e,n,t)},e.diffWordsWithSpace=a,e.formatPatch=E,e.merge=function(e,n,t){e=J(e,t),n=J(n,t);for(var r={},i=((e.index||n.index)&&(r.index=e.index||n.index),(e.newFileName||n.newFileName)&&(q(e)?q(n)?(r.oldFileName=H(r,e.oldFileName,n.oldFileName),r.newFileName=H(r,e.newFileName,n.newFileName),r.oldHeader=H(r,e.oldHeader,n.oldHeader),r.newHeader=H(r,e.newHeader,n.newHeader)):(r.oldFileName=e.oldFileName,r.newFileName=e.newFileName,r.oldHeader=e.oldHeader,r.newHeader=e.newHeader):(r.oldFileName=n.oldFileName||e.oldFileName,r.newFileName=n.newFileName||e.newFileName,r.oldHeader=n.oldHeader||e.oldHeader,r.newHeader=n.newHeader||e.newHeader)),r.hunks=[],0),o=0,l=0,a=0;i<e.hunks.length||o<n.hunks.length;){var u=e.hunks[i]||{oldStart:1/0},s=n.hunks[o]||{oldStart:1/0};if(R(u,s))r.hunks.push(U(u,l)),i++,a+=u.newLines-u.oldLines;else if(R(s,u))r.hunks.push(U(s,a)),o++,l+=s.newLines-s.oldLines;else{var f,c={oldStart:Math.min(u.oldStart,s.oldStart),oldLines:0,newStart:Math.min(u.newStart+l,s.oldStart+a),newLines:0,lines:[]},d=(f=b=y=w=m=g=v=p=h=d=void 0,c),h=u.oldStart,p=u.lines,v=s.oldStart,g=s.lines,m={offset:h,lines:p,index:0},w={offset:v,lines:g,index:0};for(Z(d,m,w),Z(d,w,m);m.index<m.lines.length&&w.index<w.lines.length;){var y=m.lines[m.index],b=w.lines[w.index];"-"!==y[0]&&"+"!==y[0]||"-"!==b[0]&&"+"!==b[0]?"+"===y[0]&&" "===b[0]?(f=d.lines).push.apply(f,k(W(m))):"+"===b[0]&&" "===y[0]?(f=d.lines).push.apply(f,k(W(w))):"-"===y[0]&&" "===b[0]?X(d,m,w):"-"===b[0]&&" "===y[0]?X(d,w,m,!0):y===b?(d.lines.push(y),m.index++,w.index++):T(d,W(m),W(w)):function(e,n,t){var r=W(n),i=W(t);if(G(r)&&G(i)){if(A(r,i)&&K(t,r,r.length-i.length))return(t=e.lines).push.apply(t,k(r));if(A(i,r)&&K(n,i,i.length-r.length))return(t=e.lines).push.apply(t,k(i))}else if(function(e,n){return e.length===n.length&&A(e,n)}(r,i))return(n=e.lines).push.apply(n,k(r));T(e,r,i)}(d,m,w)}B(d,m),B(d,w),$(d),o++,i++,r.hunks.push(c)}}return r},e.parsePatch=j,e.reversePatch=function e(n){return Array.isArray(n)?n.map(e).reverse():b(b({},n),{},{oldFileName:n.newFileName,oldHeader:n.newHeader,newFileName:n.oldFileName,newHeader:n.oldHeader,hunks:n.hunks.map(function(e){return{oldLines:e.newLines,oldStart:e.newStart,newLines:e.oldLines,newStart:e.oldStart,lines:e.lines.map(function(e){return e.startsWith("-")?"+".concat(e.slice(1)):e.startsWith("+")?"-".concat(e.slice(1)):e})}})})},e.structuredPatch=O});
