var Bose=Bose||{};Bose.Translation=function(){function a(a,b){var d,e=Bose.DataLayer.Data.Page;if(e){var f=e.formattedLocale;f&&(Granite.I18n.setUrlPrefix("/bin/i18n/b2c/dict."),Granite.I18n.setLocale("B2C."+f))}return d=b?Granite.I18n.get(a,b):Granite.I18n.get(a),c(d)}function b(a,b){var d,e=Bose.DataLayer.Data.Page;if(e){var f=e.formattedLocale;f&&(Granite.I18n.setUrlPrefix("/bin/i18n/b2b/dict."),Granite.I18n.setLocale("B2B."+f))}return d=b?Granite.I18n.get(a,b):Granite.I18n.get(a),c(d)}function c(a){return a.replace("[i]","").replace("[/i]","").replace("[b]","").replace("[/b]","")}function d(b){return b.map(a)}return{getSiteTranslation:a,getSiteTranslations:d,getB2BSiteTranslation:b,removeMarkup:c}}();