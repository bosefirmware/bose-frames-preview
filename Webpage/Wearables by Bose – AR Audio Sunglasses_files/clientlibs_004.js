var Bose=Bose||{};Bose.UserData||(Bose.UserData=function(){function a(a){var b;if(a)for(var c=0;c<a.length;c++)a[c].primary&&(b=a[c]);return b}function b(a){(f=$jq(i.userProfileGetMicroServiceUrl).val())&&a&&Bose.ajax.fetch({url:f+"/customers/"+a},function(a){a&&a.data&&d(a.data)})}function c(a,c){f=$jq(i.userProfileGetMicroServiceUrl).val();var d={gigyaUid:a.UID,uidSignature:a.UIDSignature,signatureTimestamp:a.signatureTimestamp};f&&a&&($jq.support.cors=!0,$jq.ajax({url:f+"/auth",type:"POST",async:!1,data:JSON.stringify(d),crossDomain:!0,cache:!1,headers:{"Content-Type":"text/plain"},xhrFields:{withCredentials:!0}}).done(function(d){var e=d.data;e.valid&&Bose.ajax.setCSRFToken(e.csrfToken),"populateGlobalObj"===c&&b(a.UID)}))}function d(a){Bose.UserData.PCD.primaryEmail=Bose.UserData.findPrimaryEmail(a.emailList).value,Bose.UserData.PCD.customerId=a.customerId,Bose.UserData.PCD.firstName=a.firstName,Bose.UserData.PCD.lastName=a.lastName,Bose.UserData.PCD.birthDayMonth=a.birthDayMonth,Bose.UserData.PCD.primaryPhone=a.primaryPhone,Bose.UserData.PCD.emailList=a.emailList,Bose.UserData.PCD.createdDate=a.createdDate,Bose.UserData.PCD.lastUpdated=a.lastUpdated,Bose.UserData.PCD.gigyaLastUpdated=a.gigyaLastUpdated,Bose.UserData.PCD.gigyaSynced=a.gigyaSynced,$jq(document).trigger("userData.populated")}function e(a,b){Bose.ErrorHandler.init(),a&&(Bose.UserData.gigyaUID=a.UID,c(a,b))}var f,g={UID:"",profile:{firstName:"",lastName:"",email:"",birthDay:"",birthMonth:"",gender:""},emails:"",isVerified:""},h={primaryEmail:"",customerId:"",firstName:"",lastName:"",birthDayMonth:"",primaryPhone:"",emailList:"",createdDate:"",lastUpdated:"",gigyaLastUpdated:"",gigyaSynced:!1},i={userProfileGetMicroServiceUrl:".js-userProfile-microservice-url"};return{Gigya:g,PCD:h,gigyaUID:void 0,selectors:i,populatePCDObjectInGlobalObject:d,populateNebualDataInGlobalObject:b,microServiceUrl:f,findPrimaryEmail:a,init:e}}());